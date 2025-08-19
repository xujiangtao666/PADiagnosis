
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.urls import reverse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from patient_records.models import Patient, ClinicalFeature, Doctor, PatientInfo
from .models import DiagnosisResult, DiagResult
import os
import json
from datetime import datetime, timedelta
import random
import time
import paramiko  # 用于SSH连接
import base64
import socket
from django.db.models import Q
import sys
import numpy as np

# 添加一个全局变量，用于跟踪正在进行的远程命令执行
# 格式: {patient_id: (timestamp, status, result)}
# status: 'running', 'completed', 'error'
# result: 远程执行结果，如果有
_remote_executions = {}

# 真实模型诊断函数
def real_model_diagnosis(patient_id, image_path):
    """
    使用MedCoss_inference模型进行真实的盆腹腔外伤诊断
    返回8种外伤类型的置信度
    """
    try:
        # 添加MedCoss_inference目录到Python路径
        medcoss_path = '/home/cm/code/PADiagnosis/MedCoss_inference'
        if medcoss_path not in sys.path:
            sys.path.insert(0, medcoss_path)
        
        # 导入MedCoss推理函数
        from inference_single_case import inference_single_case
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return mock_diagnosis_fallback(patient_id, image_path)
        
        # 调用MedCoss模型进行推理
        print(f"正在使用MedCoss模型对患者{patient_id}的图像进行推理: {image_path}")
        
        # 设置模型参数
        model_weights = os.path.join(medcoss_path, 'pth', 'checkpoint.pth')  # 假设权重文件路径
        if not os.path.exists(model_weights):
            # 如果权重文件不存在，查找其他可能的权重文件
            weights_dir = os.path.join(medcoss_path, 'weights')
            if os.path.exists(weights_dir):
                weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
                if weight_files:
                    model_weights = os.path.join(weights_dir, weight_files[0])
                    print(f"使用找到的权重文件: {model_weights}")
                else:
                    print("未找到模型权重文件，使用mock数据")
                    return mock_diagnosis_fallback(patient_id, image_path)
            else:
                print("权重目录不存在，使用mock数据")
                return mock_diagnosis_fallback(patient_id, image_path)
        
        # 调用推理函数
        probs = inference_single_case(
            nifti_path=image_path,
            checkpoint_path=model_weights,
            input_size=(64, 192, 192),  # 根据模型要求调整
            num_classes=8  # 8种外伤类型
        )
        
        # 确保probs是numpy数组或列表
        if isinstance(probs, np.ndarray):
            # probs的形状是(1, num_classes)，需要取第一个元素
            probs_list = probs[0].tolist()
        else:
            probs_list = probs
        
        # 构建置信度字典，与原有格式保持一致
        confidences = {}
        for i in range(8):
            if i < len(probs_list):
                confidences[f'confidence_{i}'] = float(probs_list[i])
            else:
                confidences[f'confidence_{i}'] = 0.0
        
        # 确定主要诊断结果类型
        max_confidence_key = max(confidences, key=confidences.get)
        injury_names = {
            'confidence_0': '无外伤',
            'confidence_1': '腹盆腔积血',
            'confidence_2': '肝脏损伤',
            'confidence_3': '脾脏损伤',
            'confidence_4': '右肾损伤',
            'confidence_5': '左肾损伤',
            'confidence_6': '右肾上腺损伤',
            'confidence_7': '胰腺损伤'
        }
        
        result_type = injury_names[max_confidence_key]
        
        print(f"MedCoss模型推理完成，患者{patient_id}的诊断结果: {result_type}, 置信度: {confidences}")
        
        # 添加数据来源标识
        confidences['data_source'] = 'model'
        confidences['data_source_label'] = 'AI模型推理结果'
        
        return confidences, result_type
        
    except Exception as e:
        print(f"MedCoss模型推理失败，患者{patient_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        # 如果模型推理失败，回退到mock数据
        return mock_diagnosis_fallback(patient_id, image_path)

# Mock诊断函数，用于替代远程模型调用（作为fallback）
def mock_diagnosis_fallback(patient_id, image_path):
    """
    Mock诊断函数，生成模拟的盆腹腔外伤诊断结果（作为fallback）
    返回8种外伤类型的置信度
    """
    import random
    import time
    
    print(f"开始Mock盆腹腔外伤诊断（fallback），患者ID: {patient_id}, 图像路径: {image_path}")
    
    # 模拟处理时间
    time.sleep(1)
    
    # 生成8种外伤类型的置信度
    # 0: 无外伤, 1: 腹盆腔积血, 2: 肝脏损伤, 3: 脾脏损伤
    # 4: 右肾损伤, 5: 左肾损伤, 6: 右肾上腺损伤, 7: 胰腺损伤
    
    # 随机选择1-3种可能的外伤类型
    injury_count = random.randint(0, 3)
    
    confidences = {
        'confidence_0': 0.0,  # 无外伤
        'confidence_1': 0.0,  # 腹盆腔积血
        'confidence_2': 0.0,  # 肝脏损伤
        'confidence_3': 0.0,  # 脾脏损伤
        'confidence_4': 0.0,  # 右肾损伤
        'confidence_5': 0.0,  # 左肾损伤
        'confidence_6': 0.0,  # 右肾上腺损伤
        'confidence_7': 0.0   # 胰腺损伤
    }
    
    if injury_count == 0:
        # 无外伤情况
        confidences['confidence_0'] = random.uniform(0.7, 0.95)
        # 其他类型分配剩余的小概率
        remaining = 1.0 - confidences['confidence_0']
        for i in range(1, 8):
            confidences[f'confidence_{i}'] = remaining * random.uniform(0.01, 0.2) / 7
    else:
        # 有外伤情况
        confidences['confidence_0'] = random.uniform(0.05, 0.3)
        
        # 随机选择受伤的器官
        injury_types = random.sample(range(1, 8), injury_count)
        remaining = 1.0 - confidences['confidence_0']
        
        # 为选中的外伤类型分配较高置信度
        for injury_type in injury_types:
            confidences[f'confidence_{injury_type}'] = remaining * random.uniform(0.2, 0.6) / injury_count
        
        # 为未选中的类型分配较低置信度
        used_confidence = sum(confidences[f'confidence_{i}'] for i in injury_types)
        remaining_low = remaining - used_confidence
        
        for i in range(1, 8):
            if i not in injury_types:
                confidences[f'confidence_{i}'] = remaining_low * random.uniform(0.01, 0.1) / (7 - injury_count)
    
    # 归一化确保总和为1
    total = sum(confidences.values())
    for key in confidences:
        confidences[key] = confidences[key] / total
    
    # 确定主要诊断结果类型
    max_confidence_key = max(confidences, key=confidences.get)
    injury_names = {
        'confidence_0': '无外伤',
        'confidence_1': '腹盆腔积血',
        'confidence_2': '肝脏损伤',
        'confidence_3': '脾脏损伤',
        'confidence_4': '右肾损伤',
        'confidence_5': '左肾损伤',
        'confidence_6': '右肾上腺损伤',
        'confidence_7': '胰腺损伤'
    }
    
    result_type = injury_names[max_confidence_key]
    
    print(f"Mock盆腹腔外伤诊断完成（fallback），主要结果: {result_type}, 置信度: {confidences}")
    
    # 添加数据来源标识
    confidences['data_source'] = 'mock'
    confidences['data_source_label'] = '模拟数据（仅供演示）'
    
    return confidences, result_type

# 主诊断函数，优先使用真实模型
def mock_diagnosis(patient_id, image_path):
    """
    主诊断函数，优先使用MedCoss真实模型，失败时回退到mock数据
    返回8种外伤类型的置信度
    """
    print(f"开始诊断，患者ID: {patient_id}, 图像路径: {image_path}")
    
    # 优先尝试使用真实模型
    return real_model_diagnosis(patient_id, image_path)

# 添加一个等待远程结果的函数
def wait_for_remote_result(patient_id, timeout=60, polling_interval=1):
    """
    等待远程执行结果返回
    
    参数:
    - patient_id: 患者ID
    - timeout: 最大等待时间（秒），默认60秒
    - polling_interval: 轮询间隔（秒），默认1秒
    
    返回:
    - 远程执行结果字典或None
    """
    global _remote_executions
    start_time = datetime.now()
    elapsed_time = 0
    
    print(f"开始等待患者 {patient_id} 的远程执行结果，最长等待 {timeout} 秒")
    
    while elapsed_time < timeout:
        # 检查患者的远程执行状态
        if patient_id in _remote_executions:
            timestamp, status, result = _remote_executions[patient_id]
            
            # 如果执行已完成且有结果，返回结果
            if status == 'completed' and result is not None:
                print(f"患者 {patient_id} 的远程执行结果已返回，等待时间: {elapsed_time} 秒")
                return result
            
            # 如果执行已完成但没有结果，或者执行出错，停止等待
            if status == 'completed' or status == 'error':
                print(f"患者 {patient_id} 的远程执行已结束，但未返回有效结果，状态: {status}")
                return None
        
        # 等待一段时间后再次检查
        time.sleep(polling_interval)
        elapsed_time = (datetime.now() - start_time).total_seconds()
    
    print(f"等待患者 {patient_id} 的远程执行结果超时，已等待 {timeout} 秒")
    return None

# 诊断首页视图
def diagnosis_home(request):
    # 获取搜索参数
    search_id = request.GET.get('search_id', '')
    search_image_type = request.GET.get('search_image_type', '')
    search_date = request.GET.get('search_date', '')
    
    # 初始查询：获取所有患者图像记录
    from patient_records.models import PatientInfo
    patient_images = PatientInfo.objects.all()
    
    # 应用搜索过滤
    if search_id:
        patient_images = patient_images.filter(patient_id=search_id)
    
    if search_image_type:
        patient_images = patient_images.filter(image_style__icontains=search_image_type)
    
    if search_date:
        # 按创建日期过滤
        try:
            from datetime import datetime
            search_date_obj = datetime.strptime(search_date, '%Y-%m-%d').date()
            patient_images = patient_images.filter(created_at__date=search_date_obj)
        except (ValueError, TypeError):
            # 如果日期格式不正确，忽略该过滤条件
            pass
    
    # 按创建时间倒序排序，并限制最多显示20条记录
    patient_images = patient_images.order_by('-created_at')[:20]
    
    return render(request, 'diagnosis/diagnosis_home.html', {
        'patient_images': patient_images,
        'search_id': search_id,
        'search_image_type': search_image_type,
        'search_date': search_date
    })

# 上传CT图像视图
def upload_ct(request, patient_id=None):
    """基于PatientInfo的诊断页面"""
    try:
        if patient_id:
            # 获取该患者ID的PatientInfo记录，优先获取超声图像
            from patient_records.models import PatientInfo
            patient_images = PatientInfo.objects.filter(patient_id=patient_id).order_by('-created_at')
            
            if not patient_images.exists():
                messages.error(request, f"未找到患者ID {patient_id} 的图像记录")
                return redirect('diagnosis:home')
            
            # 获取最新的PatientInfo记录作为主要参考
            patient_info = patient_images.first()
            
            # 尝试获取对应的Patient记录（如果存在）
            try:
                patient = Patient.objects.get(patient_id=patient_id)
                # 获取患者的临床特征
                try:
                    clinical_features = ClinicalFeature.objects.get(patient=patient)
                except ClinicalFeature.DoesNotExist:
                    clinical_features = None
            except Patient.DoesNotExist:
                patient = None
                clinical_features = None
                
            context = {
                'patient_info': patient_info,
                'patient_images': patient_images,
                'patient': patient,  # 可能为None
                'clinical_features': clinical_features,  # 可能为None
            }
            
            return render(request, 'diagnosis/upload_ct.html', context)
        else:
            # 如果没有提供patient_id，重定向到诊断首页
            messages.warning(request, "请先选择一位患者进行诊断")
            return redirect('diagnosis:home')
    except Exception as e:
        messages.error(request, f"获取患者信息时出错: {str(e)}")
        return redirect('diagnosis:home')

# 远程执行HUST-19模型
def run_remote_model(temp_image_path=None, patient=None, clinical_features=None, form_data=None, from_api=False):
    """
    在远程服务器上运行HUST-19模型
    
    参数:
    - patient: 患者对象（用于创建诊断结果记录）
    - from_api: 是否来自API调用
    
    返回:
    - 解析后的结果概率字典或None
    """
    global _remote_executions
    result_probabilities = None
    client = None
    
    try:
        # 如果未提供患者信息且提供了表单数据，则从表单获取患者ID
        if patient is None and form_data:
            patient_id = form_data.get('patient')
            try:
                # 优先从PatientInfo获取患者信息
                patient_info = PatientInfo.objects.get(patient_id=patient_id)
                patient = patient_info  # 使用PatientInfo作为patient对象
                
                # 尝试获取临床特征（如果Patient表中有对应记录）
                try:
                    patient_record = Patient.objects.get(patient_id=patient_id)
                    clinical_features = patient_record.clinical_features
                except Patient.DoesNotExist:
                    clinical_features = None
            except PatientInfo.DoesNotExist:
                print(f"无法获取患者信息，ID: {patient_id}")
                return None
        
        if patient is None:
            print("错误：未提供患者信息")
            return None
            
        patient_id = patient.patient_id
        
        # 检查是否有正在进行的相同患者命令执行
        current_time = datetime.now()
        execution_id = f"run_remote_model_{patient_id}_{current_time.strftime('%Y%m%d%H%M%S')}"
        print(f"开始执行ID: {execution_id} 的远程模型")
        
        if patient_id in _remote_executions:
            timestamp, status, cached_result = _remote_executions[patient_id]
            
            # 如果有已完成的结果且不超过10分钟，直接返回缓存的结果
            if status == 'completed' and cached_result is not None and (current_time - timestamp).total_seconds() < 600:
                print(f"患者 {patient_id} 已有缓存的远程执行结果，将直接返回")
                return cached_result
        
        # 标记此患者的远程执行为进行中
        print(f"标记患者 {patient_id} 的远程执行状态为进行中")
        _remote_executions[patient_id] = (current_time, 'running', None)
        
        # 远程服务器连接信息
        hostname = "202.197.33.114"
        port = 223
        username = "xuchang"
        password = "1"
        
        # HUST-19项目路径和python环境
        remote_project_path = "/data8t/xuchang/PycharmProjects/HUST-19"
        python_env = "source /data8t/xuchang/anaconda3/etc/profile.d/conda.sh && conda activate hust19"
        
        # 创建SSH客户端
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            # 连接到远程服务器
            print(f"正在连接到远程服务器 {hostname}:{port}")
            client.connect(
                hostname=hostname, 
                port=port, 
                username=username, 
                password=password,
                timeout=30,
                allow_agent=False,
                look_for_keys=False
            )
            
            # 构建命令（使用patient_id替代id）
            command = f"{python_env} && cd {remote_project_path} && python run.py --patient_id={patient.patient_id}"
            print(f"执行远程命令: {command}")
            
            # 执行命令
            stdin, stdout, stderr = client.exec_command(command, timeout=120)  # 增加超时时间到120秒
            
            # 读取输出
            stdout_str = stdout.read().decode('utf-8', errors='ignore')
            stderr_str = stderr.read().decode('utf-8', errors='ignore')
            exit_status = stdout.channel.recv_exit_status()
            
            print(f"远程执行命令完成，退出状态: {exit_status}")
            print(f"标准输出: {stdout_str}")
            if stderr_str:
                print(f"标准错误: {stderr_str}")
            
            # 尝试从标准输出中解析结果，改进正则表达式匹配
            import re
            
            # 先尝试匹配标准格式：三个浮点数以加号分隔
            result_match = re.search(r"(\d+\.\d+)\+(\d+\.\d+)\+(\d+\.\d+)", stdout_str)
            if result_match:
                probability_normal = float(result_match.group(1))
                probability_mild = float(result_match.group(2))
                probability_severe = float(result_match.group(3))
                
                print(f"从远程输出中解析到概率: 正常({probability_normal:.4f}), 轻度({probability_mild:.4f}), 重度({probability_severe:.4f})")
                
                # 创建结果概率字典
                result_probabilities = {
                    'normal': probability_normal,
                    'mild': probability_mild,
                    'severe': probability_severe
                }
                
                # 更新此患者的远程执行状态为完成，同时缓存结果
                _remote_executions[patient_id] = (current_time, 'completed', result_probabilities)
                
                print(f"执行ID: {execution_id} 的远程模型完成，结果: {result_probabilities}")
                
                # 立即关闭SSH连接
                if client:
                    client.close()
                    client = None
                    
                return result_probabilities
            else:
                # 尝试其他格式匹配，查找包含概率信息的行
                print("无法通过标准格式解析结果，尝试其他方式...")
                
                # 查找结果类型和概率
                for line in stdout_str.splitlines():
                    if "result_str" in line and "该患者" in line:
                        print(f"找到结果行: {line}")
                        # 查找下一行是否包含概率
                        break
                
                # 如果上面的方法都失败，尝试直接在整个输出中查找三个浮点数
                numbers = re.findall(r"(\d+\.\d+)", stdout_str)
                if len(numbers) >= 3:
                    print(f"在输出中找到浮点数: {numbers}")
                    probability_normal = float(numbers[0])
                    probability_mild = float(numbers[1])
                    probability_severe = float(numbers[2])
                    
                    # 检查概率和是否接近1
                    prob_sum = probability_normal + probability_mild + probability_severe
                    if 0.99 <= prob_sum <= 1.01:
                        print(f"找到有效概率值: 正常({probability_normal:.4f}), 轻度({probability_mild:.4f}), 重度({probability_severe:.4f})")
                        
                        # 创建结果概率字典
                        result_probabilities = {
                            'normal': probability_normal,
                            'mild': probability_mild,
                            'severe': probability_severe
                        }
                        
                        # 更新此患者的远程执行状态为完成，同时缓存结果
                        _remote_executions[patient_id] = (current_time, 'completed', result_probabilities)
                        
                        print(f"执行ID: {execution_id} 的远程模型完成，结果: {result_probabilities}")
                        
                        # 立即关闭SSH连接
                        if client:
                            client.close()
                            client = None
                        
                        return result_probabilities
                
                print("无法从远程输出中解析出结果概率")
                # 更新此患者的远程执行状态为完成，但没有有效结果
                _remote_executions[patient_id] = (current_time, 'completed', None)
            
        except paramiko.AuthenticationException:
            print("SSH认证失败: 用户名或密码错误")
            _remote_executions[patient_id] = (current_time, 'error', None)
        except paramiko.SSHException as e:
            print(f"SSH连接错误: {str(e)}")
            _remote_executions[patient_id] = (current_time, 'error', None)
        except socket.timeout:
            print("SSH连接超时，请检查网络或服务器状态")
            _remote_executions[patient_id] = (current_time, 'error', None)
        except socket.error as e:
            print(f"Socket错误: {str(e)}")
            _remote_executions[patient_id] = (current_time, 'error', None)
        except Exception as e:
            print(f"远程执行过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            _remote_executions[patient_id] = (current_time, 'error', None)
        
    except Exception as e:
        print(f"运行远程模型时发生错误: {str(e)}")
        # 更新状态为错误
        if patient and hasattr(patient, 'patient_id'):
            _remote_executions[patient.patient_id] = (current_time, 'error', None)
    finally:
        # 确保连接被关闭
        if client:
            try:
                client.close()
                print("SSH连接已关闭")
            except:
                pass
    
    return result_probabilities

# 解析模型输出结果
def parse_model_results(patient_id, remote_result=None):
    """
    解析模型输出的结果，优先使用远程运行的结果
    
    参数:
    - patient_id: 患者ID
    - remote_result: 远程运行得到的结果，如果提供则优先使用
    
    返回:
    - 概率字典 {'normal': x, 'mild': y, 'severe': z}
    """
    # 如果提供了远程运行结果，直接使用
    if remote_result is not None:
        print(f"使用远程运行得到的结果: {remote_result}")
        return remote_result
    
    # 尝试从多个可能的位置查找结果文件
    possible_paths = [
        f'Demo_result_{patient_id}.txt',  # 项目根目录
    ]
    
    for result_file in possible_paths:
        try:
            if os.path.exists(result_file):
                print(f"找到结果文件: {result_file}")
                import pandas as pd
                df = pd.read_csv(result_file, sep='\t')
                if len(df) > 0:
                    # 读取三个概率值
                    probability_normal = float(df['Control'].iloc[0])     # 对照组/无肺炎
                    probability_mild = float(df['Regular or Mild'].iloc[0])  # 轻度肺炎
                    probability_severe = float(df['Critically ill or Severe'].iloc[0])  # 重度肺炎
                    
                    print(f"读取到患者{patient_id}的预测结果: 正常({probability_normal:.4f}), 轻度({probability_mild:.4f}), 重度({probability_severe:.4f})")
                    
                    return {
                        'normal': probability_normal,
                        'mild': probability_mild,
                        'severe': probability_severe
                    }
        except Exception as e:
            print(f"尝试读取{result_file}时出错: {str(e)}")
    
    print(f"未找到患者{patient_id}的预测结果文件，将使用默认值")
    
    # 如果无法读取结果文件，则返回默认值（以对照组为主）
    return {
        'normal': 0.9781,  # 对照组/无肺炎
        'mild': 0.0208,    # 轻度肺炎
        'severe': 0.0011   # 重度肺炎
    }

# 检查是否已经存在最近创建的诊断记录
def check_recent_diagnosis(patient_id, time_threshold_seconds=30):
    """
    检查是否在最近时间内已经为该患者创建了诊断记录
    
    参数:
    - patient_id: 患者ID
    - time_threshold_seconds: 时间阈值（秒），默认30秒
    
    返回:
    - 如果存在返回记录对象，否则返回None
    """
    try:
        from django.utils import timezone
        import pytz
        
        # 获取当前时间，确保使用时区感知的datetime
        now = timezone.now()
        
        # 查询最近的诊断记录
        recent_diagnoses = DiagnosisResult.objects.filter(
            patient_id=patient_id,
            created_at__gte=now - timedelta(seconds=time_threshold_seconds)
        ).order_by('-created_at')
        
        if recent_diagnoses.exists():
            return recent_diagnoses.first()
        
    except Exception as e:
        print(f"检查最近诊断记录时出错: {str(e)}")
    
    return None

# 处理CT图像上传视图
def process_ct(request):
    if request.method == 'POST':
        # 获取表单数据
        patient_id = request.POST.get('patient_id')
        
        if not patient_id:
            print("请求中未提供患者ID")
            return JsonResponse({
                'success': False,
                'error': '请选择患者'
            })
        
        print(f"接收到患者 {patient_id} 的诊断请求")
        
        try:
            from django.db import transaction
            
            # 检查最近120秒内是否已经创建了诊断记录
            recent_diagnosis = check_recent_diagnosis(patient_id, 120)
            if recent_diagnosis:
                # 如果存在最近的诊断记录，使用该记录的结果
                print(f"发现患者 {patient_id} 最近的诊断记录（ID: {recent_diagnosis.id}），直接返回")
                return JsonResponse({
                    'success': True,
                    'diagnosis_id': recent_diagnosis.id,
                    'message': '使用现有的诊断结果',
                    'result_type': recent_diagnosis.result_type,
                    'confidence': recent_diagnosis.confidence * 100,
                    'probabilities': {
                        'normal': recent_diagnosis.probability_normal * 100,
                        'mild': recent_diagnosis.probability_mild * 100,
                        'severe': recent_diagnosis.probability_severe * 100
                    }
                })
            
            # 获取患者信息
            patient = get_object_or_404(Patient, patient_id=patient_id)
            
            # 上传的CT图像（可选）
            ct_image = None
            if 'ct_image' in request.FILES:
                ct_image = request.FILES['ct_image']
                print(f"成功获取CT图像: {ct_image.name}, 大小: {ct_image.size} 字节")
            else:
                print("未上传CT图像")
            
            # 检查是否有正在进行的远程执行
            global _remote_executions
            current_time = datetime.now()
            
            # 如果有进行中的执行，不要开始新的执行
            if patient_id in _remote_executions:
                timestamp, status, _ = _remote_executions[patient_id]
                if status == 'running' and (current_time - timestamp).total_seconds() < 180:  # 3分钟内
                    print(f"患者 {patient_id} 有正在进行的远程执行，将等待结果")
                    
                    # 使用一个更长的超时时间等待结果
                    print(f"等待患者 {patient_id} 的远程执行结果，最多等待180秒...")
                    probabilities = wait_for_remote_result(patient_id, timeout=180, polling_interval=3)
                    
                    if probabilities:
                        print(f"成功获取到患者 {patient_id} 的远程执行结果: {probabilities}")
                    else:
                        print(f"等待超时，未能获取患者 {patient_id} 的远程执行结果")
                        return JsonResponse({
                            'success': False,
                            'message': '远程执行超时，请稍后重试',
                            'status': 'timeout'
                        })
                else:
                    # 如果状态不是running或者已经超过3分钟，重置执行状态
                    print(f"重置患者 {patient_id} 的远程执行状态")
            
            # 如果没有获取到结果，启动新的远程执行
            if 'probabilities' not in locals() or probabilities is None:
                print(f"为患者 {patient_id} 启动新的远程执行")
                
                # 标记执行状态为running
                _remote_executions[patient_id] = (current_time, 'running', None)
                
                # 执行远程模型并等待结果
                probabilities = run_remote_model(None, patient, patient.clinical_features, None, False)
                
                # 如果仍然没有结果，再等待一次
                if probabilities is None:
                    print(f"首次执行未获取结果，再次等待患者 {patient_id} 的远程执行结果...")
                    probabilities = wait_for_remote_result(patient_id, timeout=120, polling_interval=3)
                
                # 最终验证是否有结果
                if probabilities is None:
                    print(f"多次尝试后仍未获取到患者 {patient_id} 的远程执行结果，返回错误")
                    return JsonResponse({
                        'success': False,
                        'message': '无法获取远程执行结果，请稍后重试',
                        'status': 'failed'
                    })
            
            # 确定最大概率所对应的类型
            max_prob_type = max(probabilities, key=probabilities.get)
            max_prob = probabilities[max_prob_type]
            
            # 映射回前端使用的类型
            result_type_map = {
                'normal': 'Normal',
                'mild': 'Mild',
                'severe': 'Severe'
            }
            result_type = result_type_map[max_prob_type]
            
            # 使用事务创建诊断结果记录
            with transaction.atomic():
                # 检查最近的DiagResult诊断记录
                from django.utils import timezone
                import pytz
                
                # 获取当前时间（带时区）
                current_time = timezone.now()
                time_threshold = current_time - timedelta(seconds=10)
                
                recent_diag_result = DiagResult.objects.filter(
                    patient_id=patient_id,
                    created_at__gte=time_threshold
                ).first()
                
                if recent_diag_result:
                    print(f"发现患者 {patient_id} 的最近DiagResult记录（ID: {recent_diag_result.id}），直接返回")
                    return JsonResponse({
                        'success': True,
                        'diagnosis_id': recent_diag_result.id,
                        'message': '使用等待期间创建的诊断结果',
                        'result_type': recent_diag_result.result_type,
                        'confidence': max_confidence * 100,
                        'confidences': {
                            'confidence_0': recent_diag_result.confidence_0 * 100,
                            'confidence_1': recent_diag_result.confidence_1 * 100,
                            'confidence_2': recent_diag_result.confidence_2 * 100,
                            'confidence_3': recent_diag_result.confidence_3 * 100,
                            'confidence_4': recent_diag_result.confidence_4 * 100,
                            'confidence_5': recent_diag_result.confidence_5 * 100,
                            'confidence_6': recent_diag_result.confidence_6 * 100,
                            'confidence_7': recent_diag_result.confidence_7 * 100
                        }
                    })
                
                # 创建诊断结果记录，使用真实的远程执行结果
                # 添加用户身份调试信息
                print(f"当前用户: {request.user}, 是否匿名: {request.user.is_anonymous}")
                if not request.user.is_anonymous:
                    print(f"用户ID: {request.user.id}, 用户名: {request.user.username}")
                    if hasattr(request.user, 'doctor'):
                        print(f"关联的医生ID: {request.user.doctor.doctor_id}")
                
                # 获取当前登录的医生信息
                doctor = None
                if not request.user.is_anonymous and hasattr(request.user, 'doctor'):
                    doctor = request.user.doctor
                else:
                    # 尝试从会话中获取医生ID
                    doctor_id = request.session.get('doctor_id')
                    if doctor_id:
                        try:
                            doctor = Doctor.objects.get(doctor_id=doctor_id)
                            print(f"从会话中恢复医生信息，ID: {doctor_id}")
                        except Doctor.DoesNotExist:
                            print(f"无法找到ID为{doctor_id}的医生")
                    
                    # 如果仍然无法获取医生信息，使用ID为1的医生
                    if not doctor:
                        doctor = get_object_or_404(Doctor, doctor_id=1)
                        print(f"使用默认医生，ID: 1")
                
                diagnosis_result = DiagnosisResult.objects.create(
                    patient=patient,
                    result_type=result_type,
                    confidence=max_prob,
                    probability_normal=probabilities['normal'],
                    probability_mild=probabilities['mild'],
                    probability_severe=probabilities['severe'],
                    ct_image=ct_image,
                    notes=request.POST.get('notes', ''),
                    created_by=doctor
                )
                
                print(f"成功创建患者 {patient_id} 的诊断结果记录，ID: {diagnosis_result.id}，结果: {result_type}")
            
            # 返回统一格式的响应
            return JsonResponse({
                'success': True,
                'diagnosis_id': diagnosis_result.id,
                'result_type': result_type,
                'confidence': max_prob * 100,
                'probabilities': {
                    'normal': probabilities['normal'] * 100,
                    'mild': probabilities['mild'] * 100,
                    'severe': probabilities['severe'] * 100
                },
                'message': '诊断完成'
            })
        
        except Exception as e:
            print(f"处理CT图像过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'success': False,
                'error': f'诊断过程发生错误: {str(e)}'
            })
    
    return JsonResponse({
        'success': False,
        'error': '仅支持POST请求'
    })

# 诊断结果视图
def diagnosis_result(request, diagnosis_id):
    # 获取诊断结果记录
    diagnosis = get_object_or_404(DiagResult, id=diagnosis_id)
    
    return render(request, 'diagnosis/diagnosis_result.html', {
        'diagnosis': diagnosis,
    })

# 诊断历史记录视图
def diagnosis_history(request):
    # 获取搜索参数
    search_query = request.GET.get('search_query', '')
    result_type = request.GET.get('result_type', '')
    date_range = request.GET.get('date_range', '')
    
    # 初始查询：获取所有诊断记录
    diagnosis_results = DiagResult.objects.all()
    
    # 应用搜索过滤
    if search_query:
        # 搜索患者ID或姓名
        diagnosis_results = diagnosis_results.filter(
            Q(patient__patient_id__icontains=search_query) | 
            Q(patient__name__icontains=search_query)
        )
    
    if result_type:
        # 按诊断结果类型过滤
        diagnosis_results = diagnosis_results.filter(result_type=result_type)
    
    if date_range:
        # 按日期范围过滤
        from datetime import datetime, timedelta
        now = datetime.now()
        
        if date_range == 'today':
            # 今天
            diagnosis_results = diagnosis_results.filter(created_at__date=now.date())
        elif date_range == 'week':
            # 最近一周
            one_week_ago = now - timedelta(days=7)
            diagnosis_results = diagnosis_results.filter(created_at__gte=one_week_ago)
        elif date_range == 'month':
            # 最近一个月
            one_month_ago = now - timedelta(days=30)
            diagnosis_results = diagnosis_results.filter(created_at__gte=one_month_ago)
    
    # 按创建时间倒序排序
    diagnosis_results = diagnosis_results.order_by('-created_at')
    
    # 统计各类型诊断结果的数量
    total_count = diagnosis_results.count()
    normal_count = diagnosis_results.filter(result_type='无外伤').count()
    abdominal_bleeding_count = diagnosis_results.filter(result_type='腹盆腔积血').count()
    liver_injury_count = diagnosis_results.filter(result_type='肝脏损伤').count()
    spleen_injury_count = diagnosis_results.filter(result_type='脾脏损伤').count()
    
    return render(request, 'diagnosis/diagnosis_history.html', {
        'diagnosis_results': diagnosis_results,
        'total_count': total_count,
        'normal_count': normal_count,
        'abdominal_bleeding_count': abdominal_bleeding_count,
        'liver_injury_count': liver_injury_count,
        'spleen_injury_count': spleen_injury_count,
        'search_query': search_query,
        'result_type': result_type,
        'date_range': date_range
    })

# AJAX接口：获取患者信息
def get_patient_info(request, patient_id):
    patient = get_object_or_404(Patient, patient_id=patient_id)
    
    # 获取患者的临床特征数据
    clinical_features = None
    if hasattr(patient, 'clinical_features') and patient.clinical_features:
        cf = patient.clinical_features
        clinical_features = {
            # 血液常规
            'wbc': cf.WBC,
            'rbc': cf.RBC,
            'hgb': cf.HGB,
            'hct': cf.HCT,
            'plt': cf.PLT,
            # 炎症指标
            'crp': cf.CRP,
            'esr': cf.ESR,
            'pct': cf.PCT,
            # 肝肾功能
            'alt': cf.ALT,
            'ast': cf.AST,
            'crea': cf.CREA,
            # 免疫相关
            'cd4': cf.CD4,
            'cd8': cf.CD8,
            'il6': cf.IL_6,
        }
    
    # 构建响应数据
    data = {
        'id': patient.patient_id,
        'name': patient.name,
        'gender': patient.gender,
        'age': patient.get_age(),
        'contact': patient.phone,
        'clinical_features': clinical_features,
    }
    
    return JsonResponse(data)

# AJAX接口：执行诊断
@csrf_exempt
def ajax_diagnose(request):
    """AJAX接口：执行诊断"""
    if request.method == 'POST':
        try:
            import threading
            from django.db import transaction
            
            data = json.loads(request.body)
            patient_id = data.get('patient_id')

            if not patient_id:
                return JsonResponse({'success': False, 'error': '缺少患者ID'})
            
            print(f"接收到患者 {patient_id} 的AJAX诊断请求")

            # 检查最近的诊断记录
            recent_diagnosis = check_recent_diagnosis(patient_id, 120)
            if recent_diagnosis:
                # 如果存在最近的诊断记录，使用该记录的结果
                print(f"发现患者 {patient_id} 的最近诊断记录，ID: {recent_diagnosis.id}，跳过重复创建")
                return JsonResponse({
                    'success': True,
                    'diagnosis_id': recent_diagnosis.id,
                    'result_type': recent_diagnosis.result_type,
                    'confidence': recent_diagnosis.confidence * 100,  # 转换为百分比
                    'probabilities': {
                        'normal': recent_diagnosis.probability_normal * 100,
                        'mild': recent_diagnosis.probability_mild * 100,
                        'severe': recent_diagnosis.probability_severe * 100
                    },
                    'message': '使用现有的诊断结果'
                })
            
            # 获取PatientInfo中的患者信息和图像信息
            try:
                patient_info = PatientInfo.objects.get(patient_id=patient_id)
                if not patient_info.image:
                    return JsonResponse({'success': False, 'error': f'患者{patient_id}没有关联的医学图像'})
                
                # 检查图像类型，只支持US超声诊断
                if patient_info.image_style != 'US':
                    return JsonResponse({
                        'success': False, 
                        'error': f'当前仅支持US超声诊断，患者图像类型为: {patient_info.image_style}'
                    })
                
                image_path = patient_info.image.path
                print(f"使用PatientInfo中的图像: {image_path}")
                    
            except PatientInfo.DoesNotExist:
                print(f"找不到ID为{patient_id}的患者图像信息")
                return JsonResponse({'success': False, 'error': f'找不到ID为{patient_id}的患者图像信息'})
            except Exception as e:
                print(f"获取患者数据失败: {str(e)}")
                return JsonResponse({'success': False, 'error': f'获取患者数据失败: {str(e)}'})
            
            # 检查此患者是否正在执行远程诊断
            global _remote_executions
            current_time = datetime.now()
            
            if patient_id in _remote_executions:
                timestamp, status, cached_result = _remote_executions[patient_id]
                
                # 如果已有已完成的结果且不超过10分钟，直接使用缓存的结果
                if status == 'completed' and cached_result is not None and (current_time - timestamp).total_seconds() < 600:
                    print(f"患者 {patient_id} 已有缓存的远程执行结果，将直接使用")
                    
                    # 检查cached_result的类型，如果是tuple则转换为字典
                    if isinstance(cached_result, tuple) and len(cached_result) >= 3:
                        probabilities = {
                            'normal': cached_result[0],
                            'mild': cached_result[1], 
                            'severe': cached_result[2]
                        }
                    elif isinstance(cached_result, dict):
                        probabilities = cached_result
                    else:
                        print(f"缓存结果格式不正确: {type(cached_result)}, 值: {cached_result}")
                        # 使用mock数据
                        probabilities = mock_diagnosis(patient_id, patient_info.image.path if patient_info.image else None)
                    
                    # 确定最大概率所对应的类型
                    max_prob_type = max(probabilities, key=probabilities.get)
                    max_prob = probabilities[max_prob_type]
                    
                    # 映射回前端使用的类型
                    result_type_map = {
                        'normal': 'Normal',
                        'mild': 'Mild',
                        'severe': 'Severe'
                    }
                    result_type = result_type_map[max_prob_type]
                    
                    # 使用事务创建诊断结果记录
                    with transaction.atomic():
                        # 再次检查最近的诊断记录（防止并发）
                        recent_check = check_recent_diagnosis(patient_id, 5)
                        if recent_check:
                            print(f"发现患者 {patient_id} 的最近诊断记录，ID: {recent_check.id}，使用该记录")
                            return JsonResponse({
                                'success': True,
                                'diagnosis_id': recent_check.id,
                                'result_type': recent_check.result_type,
                                'confidence': recent_check.confidence * 100,
                                'probabilities': {
                                    'normal': recent_check.probability_normal * 100,
                                    'mild': recent_check.probability_mild * 100,
                                    'severe': recent_check.probability_severe * 100
                                },
                                'message': '使用已创建的诊断结果'
                            })
                        
                        # 创建诊断结果
                        diagnosis_result = DiagnosisResult.objects.create(
                            patient=patient_info,
                            result_type=result_type,
                            confidence=max_prob,
                            probability_normal=probabilities['normal'],
                            probability_mild=probabilities['mild'],
                            probability_severe=probabilities['severe'],
                            notes=data.get('notes', ''),
                            created_by=request.user.doctor if not request.user.is_anonymous else get_object_or_404(Doctor, doctor_id=1)  # 未登录时使用ID为1的医生
                        )
                        
                        print(f"成功从缓存创建患者 {patient_id} 的诊断结果记录，ID: {diagnosis_result.id}")
                        
                        return JsonResponse({
                            'success': True,
                            'diagnosis_id': diagnosis_result.id,
                            'result_type': result_type,
                            'confidence': max_prob * 100,
                            'probabilities': {
                                'normal': probabilities['normal'] * 100,
                                'mild': probabilities['mild'] * 100,
                                'severe': probabilities['severe'] * 100
                            },
                            'message': '诊断完成（使用缓存结果）'
                        })
                
                # 如果正在执行中且不超过3分钟，等待结果
                elif status == 'running' and (current_time - timestamp).total_seconds() < 180:
                    print(f"患者 {patient_id} 已有正在进行的远程执行任务，将等待结果")
                    
                    # 等待结果最多60秒
                    probabilities = wait_for_remote_result(patient_id, timeout=60, polling_interval=3)
                    
                    if probabilities:
                        print(f"成功获取到患者 {patient_id} 的远程执行结果: {probabilities}")
                    else:
                        print(f"等待超时，未能获取患者 {patient_id} 的远程执行结果")
                        return JsonResponse({
                            'success': False,
                            'message': '远程执行超时，请稍后重试',
                            'status': 'timeout'
                        })
            
            # 如果没有获取到结果，或者没有正在执行的任务，启动新的Mock诊断
            if 'probabilities' not in locals() or probabilities is None:
                print(f"为患者 {patient_id} 启动Mock诊断")
                
                # 标记执行状态为running
                _remote_executions[patient_id] = (current_time, 'running', None)
                
                # 使用Mock诊断替代远程模型调用
                try:
                    confidences, result_type = mock_diagnosis(patient_id, image_path)
                    
                    # 更新执行状态为完成
                    _remote_executions[patient_id] = (current_time, 'completed', (confidences, result_type))
                    
                except Exception as e:
                    print(f"Mock诊断执行失败: {str(e)}")
                    _remote_executions[patient_id] = (current_time, 'error', None)
                    return JsonResponse({
                        'success': False,
                        'message': f'诊断执行失败: {str(e)}',
                        'status': 'failed'
                    })
                
                # 最终验证是否有结果
                if 'confidences' not in locals() or confidences is None:
                    print(f"Mock诊断未返回有效结果")
                    return JsonResponse({
                        'success': False,
                        'message': '诊断未返回有效结果，请稍后重试',
                        'status': 'failed'
                    })
            
            # 确定最大置信度（只考虑confidence_开头的键）
            confidence_keys = [k for k in confidences.keys() if k.startswith('confidence_')]
            max_confidence_key = max(confidence_keys, key=lambda k: confidences[k])
            max_confidence = confidences[max_confidence_key]
            
            # 使用事务创建诊断结果记录
            with transaction.atomic():
                # 再次检查最近的诊断记录（可能在等待期间由其他请求创建）
                recent_check = check_recent_diagnosis(patient_id, 10)
                if recent_check:
                    print(f"发现患者 {patient_id} 在等待期间创建的诊断记录（ID: {recent_check.id}），直接返回")
                    return JsonResponse({
                        'success': True,
                        'diagnosis_id': recent_check.id,
                        'message': '使用等待期间创建的诊断结果',
                        'result_type': recent_check.result_type,
                        'confidence': recent_check.confidence * 100,
                        'probabilities': {
                            'normal': recent_check.probability_normal * 100,
                            'mild': recent_check.probability_mild * 100,
                            'severe': recent_check.probability_severe * 100
                        }
                    })
                
                # 获取当前登录的医生信息
                doctor = None
                if not request.user.is_anonymous and hasattr(request.user, 'doctor'):
                    doctor = request.user.doctor
                else:
                    # 尝试从会话中获取医生ID
                    doctor_id = request.session.get('doctor_id')
                    if doctor_id:
                        try:
                            doctor = Doctor.objects.get(doctor_id=doctor_id)
                            print(f"从会话中恢复医生信息，ID: {doctor_id}")
                        except Doctor.DoesNotExist:
                            print(f"无法找到ID为{doctor_id}的医生")
                    
                    # 如果仍然无法获取医生信息，使用ID为1的医生
                    if not doctor:
                        doctor = get_object_or_404(Doctor, doctor_id=1)
                        print(f"使用默认医生，ID: 1")
                
                # 创建DiagResult诊断结果
                diag_result = DiagResult.objects.create(
                    patient=patient_info,
                    result_type=result_type,
                    confidence_0=confidences['confidence_0'],
                    confidence_1=confidences['confidence_1'],
                    confidence_2=confidences['confidence_2'],
                    confidence_3=confidences['confidence_3'],
                    confidence_4=confidences['confidence_4'],
                    confidence_5=confidences['confidence_5'],
                    confidence_6=confidences['confidence_6'],
                    confidence_7=confidences['confidence_7'],
                    image=patient_info.image,
                    data_source=confidences.get('data_source', 'unknown'),
                    data_source_label=confidences.get('data_source_label', '未知数据源'),
                    created_by=doctor
                )
                
                print(f"成功创建患者 {patient_id} 的DiagResult记录，ID: {diag_result.id}")
                
                return JsonResponse({
                    'success': True,
                    'diagnosis_id': diag_result.id,
                    'result_type': result_type,
                    'confidence': max_confidence * 100,
                    'confidences': {
                        'confidence_0': confidences['confidence_0'] * 100,
                        'confidence_1': confidences['confidence_1'] * 100,
                        'confidence_2': confidences['confidence_2'] * 100,
                        'confidence_3': confidences['confidence_3'] * 100,
                        'confidence_4': confidences['confidence_4'] * 100,
                        'confidence_5': confidences['confidence_5'] * 100,
                        'confidence_6': confidences['confidence_6'] * 100,
                        'confidence_7': confidences['confidence_7'] * 100
                    },
                    'data_source': confidences.get('data_source', 'unknown'),
                    'data_source_label': confidences.get('data_source_label', '未知数据源'),
                    'message': '盆腹腔外伤诊断完成'
                })

        except Exception as e:
            print(f"诊断过程发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': '仅支持POST请求'})

# 直接运行远程模型（不需要上传CT图像）
@csrf_exempt
def run_remote_diagnosis(request):
    """运行远程模型诊断，可直接从前端调用"""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': '仅支持POST请求'}, status=405)
    
    try:
        # 解析请求体
        data = json.loads(request.body)
        patient_id = data.get('patient_id')
        
        if not patient_id:
            return JsonResponse({'status': 'error', 'message': '未提供患者ID'}, status=400)
        
        # 获取患者对象
        try:
            patient = Patient.objects.get(patient_id=patient_id)
        except Patient.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': f'未找到ID为{patient_id}的患者'}, status=404)
        
        # 直接运行远程模型
        run_remote_model(patient=patient, from_api=True)
        
        # 直接返回成功响应
        return JsonResponse({
            'status': 'success',
            'message': '模型诊断请求已提交',
            'data': {
                'patient_id': patient_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
        })
    
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': '无效的JSON数据'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def diagnosis_database(request):
    """
    诊断数据库页面视图，仅渲染前端模板
    """
    return render(request, 'diagnosis/diagnosis_database.html')

def upload_patient_images(request):
    """上传患者图像页面和处理逻辑"""
    if request.method == 'POST':
        try:
            # 处理文件上传
            uploaded_files = request.FILES.getlist('images')
            image_type = request.POST.get('image_type', 'CT')
            
            if not uploaded_files:
                messages.error(request, '请选择要上传的图像文件')
                return redirect('diagnosis:upload_patient_images')
            
            from patient_records.models import PatientInfo
            from django.db import transaction
            
            # 获取当前登录的医生
            doctor_id = request.session.get('doctor_id')
            if not doctor_id:
                messages.error(request, '请先登录')
                return redirect('login')
            
            doctor = get_object_or_404(Doctor, doctor_id=doctor_id)
            
            success_count = 0
            error_files = []
            
            with transaction.atomic():
                for uploaded_file in uploaded_files:
                    try:
                        # 检查文件格式
                        if not uploaded_file.name.lower().endswith('.nii.gz'):
                            error_files.append(f"{uploaded_file.name}: 不支持的文件格式，只支持.nii.gz格式")
                            continue
                        
                        # 从文件名中提取患者ID
                        file_name = uploaded_file.name
                        patient_id = extract_patient_id_from_filename(file_name)
                        
                        if not patient_id:
                            error_files.append(f"{uploaded_file.name}: 无法从文件名中提取患者ID")
                            continue
                        
                        # 确定存储路径
                        import os
                        from django.conf import settings
                        
                        # 获取图像类型对应的文件夹
                        type_folder_map = {
                            'US': 'US',
                            'CT': 'CT', 
                            'MRI': 'MRI',
                            'X-ray': 'X-ray'
                        }
                        folder_name = type_folder_map.get(image_type, 'CT')
                        
                        # 创建目标目录
                        upload_dir = os.path.join('data', 'upload', folder_name)
                        full_upload_dir = os.path.join(settings.BASE_DIR, upload_dir)
                        os.makedirs(full_upload_dir, exist_ok=True)
                        
                        # 构建文件路径
                        file_path = os.path.join(upload_dir, uploaded_file.name)
                        full_file_path = os.path.join(full_upload_dir, uploaded_file.name)
                        
                        # 保存文件到指定目录
                        with open(full_file_path, 'wb+') as destination:
                            for chunk in uploaded_file.chunks():
                                destination.write(chunk)
                        
                        # 创建PatientInfo记录
                        patient_info = PatientInfo.objects.create(
                            patient_id=patient_id,
                            image_style=image_type,
                            image=file_path,
                            created_by=doctor
                        )
                        
                        success_count += 1
                        
                    except Exception as e:
                        error_files.append(f"{uploaded_file.name}: {str(e)}")
                        continue
            
            # 显示结果消息
            if success_count > 0:
                messages.success(request, f'成功上传 {success_count} 个图像文件')
            
            if error_files:
                for error in error_files:
                    messages.error(request, error)
            
            return redirect('diagnosis:upload_patient_images')
            
        except Exception as e:
            messages.error(request, f'上传过程中发生错误: {str(e)}')
            return redirect('diagnosis:upload_patient_images')
    
    # GET请求，显示上传页面
    return render(request, 'diagnosis/upload_patient_images.html')

def extract_patient_id_from_filename(filename):
    """从文件名中提取患者ID"""
    import re
    
    # 移除文件扩展名
    name_without_ext = filename.replace('.nii.gz', '')
    
    # 尝试多种模式提取患者ID
    patterns = [
        r'patient[_-]?(\d+)',  # patient_123, patient-123, patient123
        r'(\d+)',              # 直接的数字
        r'ID[_-]?(\d+)',       # ID_123, ID-123, ID123
        r'P(\d+)',             # P123
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name_without_ext, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None

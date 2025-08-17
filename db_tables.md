# 盆腹腔智能诊断系统数据库表结构与接口设计

## 项目概述

盆腹腔智能诊断系统是一个基于Django的多模态医疗诊断平台，集成了CT图像诊断、X光图像分割和临床特征分析功能。系统采用模块化设计，包含三个核心应用：患者记录管理、诊断分析和图像分割。

## 数据库表结构

### 1. 医生表(Doctor)

| 字段名 | 数据类型 | 说明 | 约束 |
|--------|----------|------|------|
| doctor_id | BigAutoField | 医生ID | 主键 |
| password | CharField(128) | 密码 | 非空 |
| email | EmailField(100) | 邮箱 | 非空 |
| full_name | CharField(30) | 全名 | 非空 |
| created_at | DateTimeField | 创建时间 | 自动添加 |
| last_login | DateTimeField | 最后登录时间 | 自动更新 |

**表名**: `doctor`

### （弃用）2. 患者基本信息表(Patient)

| 字段名 | 数据类型 | 说明 | 约束 |
|--------|----------|------|------|
| patient_id | BigAutoField | 患者ID | 主键 |
| name | CharField(30) | 姓名 | 非空 |
| gender | CharField(10) | 性别 | 非空(选项) |
| birth_date | DateField | 出生日期 | 非空 |
| id_card | CharField(18) | 身份证号 | 可空 |
| phone | CharField(11) | 联系电话 | 可空 |
| emergency_contact | CharField(30) | 紧急联系人 | 可空 |
| emergency_phone | CharField(11) | 紧急联系电话 | 可空 |
| created_at | DateTimeField | 创建时间 | 自动添加 |
| updated_at | DateTimeField | 更新时间 | 自动更新 |
| created_by | ForeignKey | 创建人 | 外键(Doctor) |

**表名**: `patient_basic_info`

### 3. 患者医学图像信息表(PatientInfo)

| 字段名 | 数据类型 | 说明 | 约束 |
|--------|----------|------|------|
| patient_id | BigAutoField | 患者ID | 主键 |
| image_style | CharField(30) | 医学图像类别 | 非空 |
| image | ImageField | 医学图像 | 非空 |
| created_at | DateTimeField | 创建时间 | 自动添加 |
| updated_at | DateTimeField | 更新时间 | 自动更新 |
| created_by | ForeignKey | 创建人 | 外键(Doctor) |

**表名**: `patient_image_info`

### （弃用）4. 临床特征表(ClinicalFeature)

| 字段名 | 数据类型 | 说明 | 约束 |
|--------|----------|------|------|
| patient | OneToOneField | 患者 | 主键/外键(Patient) |
| age | CharField(10) | 年龄 | 非空 |
| gender | CharField(10) | 性别 | 非空 |
| body_temperature | CharField(10) | 体温 | 非空 |
| underlying_diseases | TextField | 基础疾病 | 可空 |

**临床指标分类**:

| 指标类别 | 字段数量 | 包含指标示例 |
|--------|----------|------------|
| 常规血液检查 | 21项 | MCHC, MCH, MCV, HCT, HGB, RBC, WBC等 |
| 凝血功能检查 | 6项 | DD, TT, FIB, APTT, INR, PT等 |
| 炎症指标 | 3项 | ESR, CRP, PCT |
| 生化指标 | 42项 | ALB, ALT, AST, TBIL, GLU, CHOL等 |
| 免疫细胞分型 | 6项 | CD3, CD4, CD8, BC, NKC, CD4_CD8等 |
| 细胞因子 | 6项 | IL_2, IL_4, IL_6, IL_10, TNF, IFN等 |
| 尿常规指标 | 13项 | URBC, UWBC, SG, PH, BACT等 |
| 其他生化指标 | 31项 | IBIL, AT_III, LAC, RF, HC等 |

**表名**: `clinical_feature`

### （弃用）5. 诊断结果表(DiagnosisResult)

| 字段名 | 数据类型 | 说明 | 约束 |
|--------|----------|------|------|
| id | AutoField | 诊断结果ID | 主键 |
| patient | ForeignKey | 患者 | 外键(PatientInfo) |
| result_type | CharField(20) | 诊断结果类型 | 非空(选项) |
| confidence | FloatField | 置信度 | 非空 |
| probability_normal | FloatField | 无肺炎概率 | 默认0.0 |
| probability_mild | FloatField | 轻度肺炎概率 | 默认0.0 |
| probability_severe | FloatField | 重度肺炎概率 | 默认0.0 |
| ct_image | ImageField | CT图像 | 可空 |
| notes | TextField | 临床备注 | 可空 |
| created_at | DateTimeField | 诊断时间 | 自动添加 |
| updated_at | DateTimeField | 更新时间 | 自动更新 |
| created_by | ForeignKey | 创建医生 | 外键(Doctor) |

**诊断结果类型选项**:
- Normal: 无肺炎
- Mild: 轻度肺炎  
- Severe: 重度肺炎

**表名**: `diagnosis_diagnosisresult`

### 6. 多标签诊断结果表(DiagResult)

| 字段名 | 数据类型 | 说明 | 约束 |
|--------|----------|------|------|
| id | AutoField | 诊断结果ID | 主键 |
| result_type | CharField(20) | 诊断结果标签 | 非空 |
| patient | ForeignKey | 患者 | 外键(PatientInfo) |
| confidence_0 | FloatField | 无外伤置信度 | 默认0.0 |
| confidence_1 | FloatField | 腹盆腔或腹膜后积血/血肿置信度 | 默认0.0 |
| confidence_2 | FloatField | 肝脏损伤置信度 | 默认0.0 |
| confidence_3 | FloatField | 脾脏损伤置信度 | 默认0.0 |
| confidence_4 | FloatField | 右肾损伤置信度 | 默认0.0 |
| confidence_5 | FloatField | 左肾损伤置信度 | 默认0.0 |
| confidence_6 | FloatField | 右肾上腺损伤置信度 | 默认0.0 |
| confidence_7 | FloatField | 胰腺损伤置信度 | 默认0.0 |
| image | ImageField | CT图像 | 非空 |
| created_at | DateTimeField | 诊断时间 | 自动添加 |
| updated_at | DateTimeField | 更新时间 | 自动更新 |
| created_by | ForeignKey | 创建医生 | 外键(Doctor) |

**表名**: `diagnosis_diagresult`

### （弃用）7. 分割结果表(SegmentationResult)

| 字段名 | 数据类型 | 说明 | 约束 |
|--------|----------|------|------|
| id | AutoField | 分割结果ID | 主键 |
| patient | ForeignKey | 患者 | 外键(Patient) |
| diagnosis | ForeignKey | 关联诊断结果 | 外键(DiagnosisResult) |
| xray_image | ImageField | X光图像 | 非空 |
| segmentation_image | ImageField | 分割结果图像 | 可空 |
| prompt_text | TextField | 分割提示文本 | 可空 |
| created_at | DateTimeField | 创建时间 | 自动添加 |
| updated_at | DateTimeField | 更新时间 | 自动更新 |
| created_by | ForeignKey | 创建医生 | 外键(Doctor) |

**表名**: `segmentation_segmentationresult`

## 数据库表间关系

| 关系类型 | 相关表 | 关系描述 |
|--------|----------|------------|
| 一对多 | Doctor → Patient | 一个医生可以创建多个患者记录 |
| 一对多 | Doctor → PatientInfo | 一个医生可以创建多个患者图像记录 |
| 一对多 | Doctor → ClinicalFeature | 一个医生可以创建多份临床特征记录 |
| 一对多 | Doctor → DiagnosisResult | 一个医生可以创建多个诊断结果 |
| 一对多 | Doctor → DiagResult | 一个医生可以创建多个多标签诊断结果 |
| 一对多 | Doctor → SegmentationResult | 一个医生可以创建多个分割结果 |
| 一对一 | Patient → ClinicalFeature | 一个患者对应一份临床特征数据 |
| 一对多 | Patient → SegmentationResult | 一个患者可以有多次分割记录 |
| 一对多 | PatientInfo → DiagnosisResult | 一个患者图像可以有多次诊断记录 |
| 一对多 | PatientInfo → DiagResult | 一个患者图像可以有多次多标签诊断记录 |
| 一对多 | DiagnosisResult → SegmentationResult | 一次诊断可以对应多次分割 |

## 接口设计

### 1. 用户认证接口

| 接口路径 | HTTP方法 | 功能描述 | 依赖表 |
|---------|----------|----------|--------|
| `/` | GET | 登录页面 | Doctor |
| `/login/` | GET/POST | 用户登录 | Doctor |
| `/register/` | GET/POST | 用户注册 | Doctor |
| `/logout/` | GET | 用户登出 | Doctor |
| `/profile/` | GET | 医生资料 | Doctor |
| `/forget-password/` | GET/POST | 忘记密码 | Doctor |
| `/verify-code/` | GET/POST | 验证码验证 | Doctor |
| `/reset-password/` | GET/POST | 重置密码 | Doctor |

### 2. 患者记录管理接口

| 接口路径 | HTTP方法 | 功能描述 | 依赖表 |
|---------|----------|----------|--------|
| `/patient-records/` | GET | 患者列表 | Patient, Doctor |
| `/patient-records/add/` | GET/POST | 新增患者 | Patient, Doctor |
| `/patient-records/<id>/` | GET | 患者详情 | Patient, ClinicalFeature, Doctor |
| `/patient-records/<id>/edit/` | GET/POST | 编辑患者 | Patient, Doctor |
| `/patient-records/<id>/delete/` | POST | 删除患者 | Patient, Doctor |
| `/patient-records/<id>/add-record/` | GET/POST | 添加就诊记录 | ClinicalFeature, Doctor |

### 3. 诊断分析接口

| 接口路径 | HTTP方法 | 功能描述 | 依赖表 |
|---------|----------|----------|--------|
| `/diagnosis/` | GET | 诊断首页 | Patient, DiagnosisResult, Doctor |
| `/diagnosis/upload-ct/` | GET/POST | 上传CT图像 | PatientInfo, Doctor |
| `/diagnosis/upload-ct/<id>/` | GET/POST | 为指定患者上传CT | PatientInfo, Doctor |
| `/diagnosis/process-ct/` | POST | 处理CT图像 | PatientInfo, DiagnosisResult, Doctor |
| `/diagnosis/result/<id>/` | GET | 诊断结果 | DiagnosisResult, PatientInfo, Doctor |
| `/diagnosis/history/` | GET | 诊断历史 | DiagnosisResult, PatientInfo, Doctor |
| `/diagnosis/database/` | GET | 诊断数据库 | DiagnosisResult, PatientInfo, Doctor |

**AJAX接口**:
| 接口路径 | HTTP方法 | 功能描述 | 依赖表 |
|---------|----------|----------|--------|
| `/diagnosis/api/patient-info/<id>/` | GET | 获取患者信息 | Patient, ClinicalFeature |
| `/diagnosis/api/diagnose/` | POST | AJAX诊断 | PatientInfo, DiagnosisResult |
| `/diagnosis/api/remote-diagnose/` | POST | 远程诊断 | PatientInfo, DiagnosisResult |

### 4. 图像分割接口

| 接口路径 | HTTP方法 | 功能描述 | 依赖表 |
|---------|----------|----------|--------|
| `/segmentation/` | GET | 分割首页 | Patient, SegmentationResult, Doctor |
| `/segmentation/upload-xray/` | GET/POST | 上传X光图像 | Patient, SegmentationResult, Doctor |
| `/segmentation/upload-xray/<id>/` | GET/POST | 为指定患者上传X光 | Patient, SegmentationResult, Doctor |
| `/segmentation/result/<id>/` | GET | 分割结果 | SegmentationResult, Patient, Doctor |
| `/segmentation/history/` | GET | 分割历史 | SegmentationResult, Patient, Doctor |
| `/segmentation/history/<id>/` | GET | 指定患者分割历史 | SegmentationResult, Patient, Doctor |

**API接口**:
| 接口路径 | HTTP方法 | 功能描述 | 依赖表 |
|---------|----------|----------|--------|
| `/segmentation/api/upload-xray/` | POST | X光图像上传API | Patient, SegmentationResult, Doctor |
| `/segmentation/api/deepseek-chat/` | POST | DeepSeek聊天接口 | SegmentationResult |

## 数据流动过程

### 1. 用户认证流程

```
用户访问 → 登录验证 → 会话创建 → 权限验证 → 功能访问
    ↓
Doctor表查询 → 密码验证 → 会话存储 → 中间件检查 → 视图渲染
```

### 2. 患者记录管理流程

```
医生登录 → 患者列表 → 新增/编辑患者 → 保存患者信息 → 临床特征录入
    ↓
Patient表操作 → 表单验证 → 数据持久化 → ClinicalFeature关联 → 完整病历
```

### 3. 诊断分析流程

```
选择患者 → 上传CT图像 → 图像预处理 → AI模型分析 → 结果存储 → 结果展示
    ↓
PatientInfo创建 → 图像存储 → 远程诊断 → DiagnosisResult保存 → 模板渲染
```

**详细流程**:
1. **患者选择**: 从Patient表查询患者信息
2. **图像上传**: 创建PatientInfo记录，存储CT图像
3. **远程诊断**: 通过SSH连接远程服务器执行诊断
4. **结果处理**: 解析诊断结果，保存到DiagnosisResult表
5. **结果展示**: 渲染诊断结果页面，显示置信度和概率

### 4. 图像分割流程

```
选择患者 → 上传X光图像 → 输入分割提示 → AI分割处理 → 结果存储 → 结果展示
    ↓
Patient关联 → 图像存储 → 文本处理 → 分割算法 → SegmentationResult保存 → 可视化展示
```

**详细流程**:
1. **患者关联**: 从Patient表获取患者信息
2. **图像上传**: 存储X光图像到指定目录
3. **提示处理**: 医生输入分割提示文本
4. **AI分割**: 调用DeepSeek API进行图像分割
5. **结果存储**: 保存分割结果图像和相关信息
6. **结果展示**: 显示原图和分割结果对比

### 5. 数据查询流程

```
用户请求 → URL路由 → 视图函数 → 模型查询 → 数据库操作 → 结果返回
    ↓
权限验证 → 参数解析 → ORM查询 → SQL执行 → 数据序列化 → 响应渲染
```

## 技术架构

### 1. 后端框架
- **Django 4.2**: 主Web框架
- **Django ORM**: 数据库操作
- **Django Admin**: 后台管理

### 2. 数据库
- **MySQL**: 主数据库
- **Django Migrations**: 数据库版本管理

### 3. 文件存储
- **本地文件系统**: 医学图像存储
- **ImageField**: Django图像字段处理

### 4. 外部集成
- **SSH连接**: 远程诊断服务器
- **DeepSeek API**: AI图像分割服务
- **Paramiko**: Python SSH客户端

### 5. 前端技术
- **HTML/CSS/JavaScript**: 基础前端
- **Bootstrap**: UI框架
- **AJAX**: 异步数据交互

## 系统特点

1. **多模态诊断**: 支持CT图像和X光图像分析
2. **AI集成**: 集成远程AI诊断和图像分割服务
3. **临床数据**: 128项临床指标支持
4. **模块化设计**: 三个独立应用，职责清晰
5. **权限管理**: 基于医生的访问控制
6. **数据完整性**: 完整的患者-诊断-分割数据链路

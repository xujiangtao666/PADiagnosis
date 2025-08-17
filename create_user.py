#!/usr/bin/env python
"""
盆腹腔诊断系统用户创建脚本
支持创建医生用户和管理员用户
"""

import os
import sys
import re
import getpass
from pathlib import Path

# 添加项目路径到Python路径
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Demo.settings')

import django
django.setup()

from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.utils import timezone
from patient_records.models import Doctor


def validate_doctor_id(doctor_id):
    """验证医生ID格式"""
    if not re.match(r'^\d{10}$', doctor_id):
        return False, "医生ID必须是10位数字"
    
    if Doctor.objects.filter(doctor_id=doctor_id).exists():
        return False, "此医生ID已被注册"
    
    return True, ""


def validate_password(password):
    """验证密码强度"""
    if len(password) < 10:
        return False, "密码长度至少10位"
    
    if not re.search(r'[a-z]', password):
        return False, "密码必须包含小写字母"
    
    if not re.search(r'[A-Z]', password):
        return False, "密码必须包含大写字母"
    
    if not re.search(r'\d', password):
        return False, "密码必须包含数字"
    
    return True, ""


def validate_email_address(email):
    """验证邮箱格式"""
    try:
        validate_email(email)
        return True, ""
    except ValidationError:
        return False, "请输入有效的邮箱地址"


def create_doctor_user():
    """创建医生用户"""
    print("\n=== 创建医生用户 ===")
    
    while True:
        doctor_id = input("请输入医生ID（10位数字）: ").strip()
        valid, error = validate_doctor_id(doctor_id)
        if valid:
            break
        print(f"错误: {error}")
    
    while True:
        full_name = input("请输入医生全名: ").strip()
        if len(full_name) >= 2:
            break
        print("错误: 全名至少需要2个字符")
    
    while True:
        email = input("请输入邮箱地址: ").strip()
        valid, error = validate_email_address(email)
        if not valid:
            print(f"错误: {error}")
            continue
        
        # 检查邮箱是否已存在
        if Doctor.objects.filter(email=email).exists():
            print("错误: 此邮箱已被注册")
            continue
        
        break
    
    while True:
        password = getpass.getpass("请输入密码（至少10位，包含大小写字母和数字）: ")
        valid, error = validate_password(password)
        if not valid:
            print(f"错误: {error}")
            continue
        
        confirm_password = getpass.getpass("请确认密码: ")
        if password != confirm_password:
            print("错误: 两次输入的密码不一致")
            continue
        
        break
    
    try:
        # 创建医生用户
        doctor = Doctor(
            doctor_id=doctor_id,
            full_name=full_name,
            email=email,
            password=make_password(password),
            created_at=timezone.now()
        )
        doctor.save()
        
        print(f"\n✅ 医生用户创建成功！")
        print(f"医生ID: {doctor_id}")
        print(f"全名: {full_name}")
        print(f"邮箱: {email}")
        print(f"创建时间: {doctor.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ 创建医生用户失败: {str(e)}")


def create_admin_user():
    """创建管理员用户（Django User）"""
    print("\n=== 创建管理员用户 ===")
    
    while True:
        username = input("请输入管理员用户名: ").strip()
        if len(username) >= 3:
            if not User.objects.filter(username=username).exists():
                break
            else:
                print("错误: 此用户名已存在")
        else:
            print("错误: 用户名至少需要3个字符")
    
    while True:
        email = input("请输入邮箱地址: ").strip()
        valid, error = validate_email_address(email)
        if not valid:
            print(f"错误: {error}")
            continue
        
        # 检查邮箱是否已存在
        if User.objects.filter(email=email).exists():
            print("错误: 此邮箱已被注册")
            continue
        
        break
    
    while True:
        password = getpass.getpass("请输入密码（至少10位，包含大小写字母和数字）: ")
        valid, error = validate_password(password)
        if not valid:
            print(f"错误: {error}")
            continue
        
        confirm_password = getpass.getpass("请确认密码: ")
        if password != confirm_password:
            print("错误: 两次输入的密码不一致")
            continue
        
        break
    
    try:
        # 创建管理员用户
        user = User.objects.create_superuser(
            username=username,
            email=email,
            password=password
        )
        
        print(f"\n✅ 管理员用户创建成功！")
        print(f"用户名: {username}")
        print(f"邮箱: {email}")
        print(f"创建时间: {user.date_joined.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"管理员权限: 是")
        print(f"可以访问Django管理后台: http://127.0.0.1:8000/admin/")
        
    except Exception as e:
        print(f"\n❌ 创建管理员用户失败: {str(e)}")


def main():
    """主函数"""
    print("=" * 50)
    print("     肺炎诊断系统用户创建工具")
    print("=" * 50)
    
    while True:
        print("\n请选择要创建的用户类型：")
        print("1. 医生用户（用于系统登录和患者管理）")
        print("2. 管理员用户（用于Django后台管理）")
        print("3. 退出")
        
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == '1':
            create_doctor_user()
        elif choice == '2':
            create_admin_user()
        elif choice == '3':
            print("\n再见！")
            break
        else:
            print("无效选择，请输入 1、2 或 3")
        
        # 询问是否继续
        if choice in ['1', '2']:
            continue_choice = input("\n是否继续创建其他用户？(y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', '是']:
                print("\n再见！")
                break


if __name__ == "__main__":
    main()
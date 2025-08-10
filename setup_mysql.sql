-- 创建数据库
CREATE DATABASE IF NOT EXISTS test CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 创建用户
CREATE USER IF NOT EXISTS 'django_user'@'localhost' IDENTIFIED BY 'onlyone991114';

-- 授权用户对test数据库的所有权限
GRANT ALL PRIVILEGES ON test.* TO 'django_user'@'localhost';

-- 刷新权限
FLUSH PRIVILEGES;

-- 显示创建的数据库
SHOW DATABASES;

-- 显示用户权限
SHOW GRANTS FOR 'django_user'@'localhost';

-- 选择test数据库并显示字符集设置
USE test;
SELECT @@character_set_database, @@collation_database;

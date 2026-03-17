---
name: aliyunpan
description: 阿里云盘命令行客户端工具（v0.3.x），用于管理阿里云盘文件。触发场景包括：(1) 用户提到阿里云盘或 aliyunpan，(2) 需要上传/下载云盘文件，(3) 管理云盘文件和目录（列出、创建、删除、移动、重命名），(4) 分享云盘文件，(5) 同步备份本地和云盘文件，(6) 管理阿里云盘账号，(7) 操作共享相册。支持文件浏览、传输、管理、分享和自动化同步备份。
---

# aliyunpan - 阿里云盘命令行客户端

> **版本说明**：本文档基于 aliyunpan v0.3.7 编写，适用于 v0.3.x 系列版本。不同版本的功能可能有差异。

## Overview

aliyunpan 是阿里云盘的命令行客户端工具，提供类似 Linux shell 的文件操作体验。支持文件的浏览、上传、下载、管理、分享和自动化同步备份功能。

**核心功能**：
- 文件浏览与导航（ls, cd, pwd, tree）
- 文件传输（upload, download）
- 文件管理（mkdir, rm, mv, rename, cp）
- 文件分享（创建分享链接）
- 同步备份（本地→云盘、云盘→本地）
- 多账号管理（login, logout, su）
- 配置管理（config）
- 共享相册操作（album）

## Quick Start

### 登录账号

首次使用需要登录阿里云盘账号。aliyunpan 使用基于浏览器的授权方式：

```bash
# 启动交互模式
aliyunpan

# 在交互模式中登录
login
```

按照提示在浏览器中完成：
1. 官方 API 授权登录
2. Web 界面扫码登录

登录成功后会自动保存凭证，下次使用无需重新登录。

### 基本配置

设置下载保存目录：

```bash
# 在交互模式中
config set -savedir /path/to/download

# 或直接执行
aliyunpan config set -savedir /path/to/download
```

查看当前配置：

```bash
config
```

### 重要：环境变量配置

**在使用 aliyunpan 之前，必须先清除所有代理相关的环境变量**，否则可能导致连接失败或网络问题。

```bash
# 清除代理环境变量（每次使用前执行）
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY

# 然后再启动 aliyunpan
aliyunpan
```

**推荐做法**：创建一个启动脚本或别名：

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
alias aliyunpan='unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY && /usr/local/bin/aliyunpan'
```

或者创建启动脚本：

```bash
#!/bin/bash
# ~/bin/aliyunpan-start.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
/usr/local/bin/aliyunpan "$@"
```

**注意**：
- 代理环境变量会影响 aliyunpan 的网络连接
- 必须在每次使用 aliyunpan 前清除这些变量
- 如果使用 aliyunpan 内置的代理功能，请使用 `config set -proxy` 命令设置

### 使用模式

**交互模式**（推荐）：

```bash
aliyunpan
# 进入交互式 shell，可以连续执行多个命令
```

**命令行模式**：

```bash
# 直接执行单个命令
aliyunpan ls /docs
aliyunpan download /file.pdf
aliyunpan upload /local/file.txt /cloud/dir
```

## 文件浏览与导航

### 基本命令

```bash
# 显示当前工作目录
pwd

# 切换目录
cd /docs              # 切换到 /docs
cd ..                 # 返回上级目录
cd /                  # 返回根目录

# 列出目录内容
ls                    # 列出当前目录
ls /path/to/dir      # 列出指定目录
ll                    # 详细列表模式
l                     # 简化别名

# 显示目录树
tree                  # 显示当前目录树
tree /docs           # 显示指定目录树
tree -fp /docs       # 显示文件完整绝对路径
tree -fs /docs       # 显示文件大小
```

**注意**：
- 路径必须使用正斜杠 `/`，不是反斜杠 `\`
- 包含空格的路径需要用引号包围：`cd "我的文件夹"`

### 查看文件信息

```bash
# 详细列表模式显示文件大小、修改时间等信息
ll

# 查看存储空间使用情况
quota
```

## 文件上传

### 基本上传

```bash
# 上传单个文件
upload <本地路径> <云盘目录>
u <本地路径> <云盘目录>          # 简写形式

# 示例
upload /home/user/document.pdf /docs
upload ./photo.jpg /相册/2024
u /local/file.txt /backup
```

### 上传多个文件

```bash
# 上传多个文件到同一目录
upload <文件1> <文件2> <文件3> <云盘目录>

# 上传整个目录
upload /home/user/photos /相册/2024
```

### 高级上传选项

```bash
# 排除特定文件（使用正则表达式）
upload -exn ".*\.tmp$" /local/path /cloud/path
upload -exn ".*\.log$" /local/logs /backup/logs

# 示例：排除所有临时文件和日志文件
upload -exn ".*\.(tmp|log)$" /project /backup
```

**上传注意事项**：
- 上传大文件可能需要较长时间
- 确保有足够的云盘存储空间
- 支持断点续传

## 文件下载

### 基本下载

```bash
# 下载单个文件
download <云盘文件路径>
d <云盘文件路径>                # 简写形式

# 示例
download /docs/report.pdf
download IMG_0106.JPG
d /videos/movie.mp4
```

### 下载目录

```bash
# 下载整个目录
download /docs/project
download /相册/2024
```

### 高级下载选项

```bash
# 覆盖已存在的文件
download --ow <文件>

# 跳过同名文件
download --skip <文件>

# 指定并行线程数（提高下载速度）
download -p 10 <文件>

# 多用户联合下载（适用于大文件）
download --md <文件>
```

### 下载配置

```bash
# 设置下载保存目录
config set -savedir /path/to/download

# 设置最大下载并发数
config set -max_download_parallel 10
```

**下载速度优化**：
- 如果速度不稳定，尝试设置并发数为 1：`config set -max_download_parallel 1`
- 需要开通"三方应用权益包"才能享受加速下载
- 实际速度受网络环境和会员类型影响

**重要提示**：
- 默认保存位置：程序目录下的 `download/` 文件夹
- 重名文件默认自动跳过

## 文件管理

### 创建目录

```bash
# 创建目录
mkdir <目录名>

# 示例
mkdir /docs/2024
mkdir "新建文件夹"
mkdir /backup/important
```

### 删除文件和目录

```bash
# 删除文件或目录（可在回收站中恢复）
rm <路径>

# 示例
rm file.txt
rm /docs/old_folder
rm /backup/obsolete.zip
```

**注意**：
- 删除的文件会进入回收站，可以在网页端恢复
- 删除多个文件时，确保所有路径都存在

### 移动和重命名

```bash
# 移动文件或目录
mv <源路径> <目标路径>

# 示例
mv /docs/old.txt /archive/old.txt
mv /photos/2023 /archive/photos_2023

# 重命名（必须在同一目录内）
rename <旧名称> <新名称>

# 示例
rename old_name.txt new_name.txt
rename "旧文件夹" "新文件夹"
```

**重要限制**：
- `rename` 命令只能在同一目录内使用
- 跨目录重命名请使用 `mv` 命令

### 复制文件

```bash
# 复制文件或目录
cp <源路径> <目标路径>

# 示例
cp /docs/file.txt /backup/file.txt
cp /photos/2024 /backup/photos_2024
```

## 文件分享

### 创建分享链接

```bash
# 创建分享链接
share set <文件或目录路径>
share s <文件或目录路径>        # 简写

# 示例
share set /docs/report.pdf
share set IMG_0106.JPG
share s /videos/movie.mp4
```

### 分享模式

aliyunpan 支持三种分享模式：

```bash
# 模式 1：私密分享（需要提取码）
share set -mode 1 <文件>

# 模式 2：公开分享（无需提取码）
share set -mode 2 <文件>

# 模式 3：快传（特殊分享方式）
share set -mode 3 <文件>
```

**分享模式说明**：
- **模式 1（私密）**：生成需要提取码的分享链接，安全性较高
- **模式 2（公开）**：生成无需提取码的分享链接，方便分享
- **模式 3（快传）**：特殊的分享方式，具体特性请参考官方文档

**示例**：
```bash
# 创建私密分享
share set -mode 1 /docs/confidential.pdf

# 创建公开分享
share set -mode 2 /photos/vacation.jpg

# 使用快传
share set -mode 3 /videos/movie.mp4
```

## 同步备份

aliyunpan 提供强大的同步备份功能，支持两种模式：

### 同步模式

1. **upload（上传模式）**：备份本地文件到云盘
2. **download（下载模式）**：备份云盘文件到本地

### 备份策略

- **exclusive（排他备份）**：目标目录多余的文件会被删除，实现一对一镜像备份
- **increment（增量备份）**：目标目录多余的文件不会被删除，仅添加新文件

### 同步命令

```bash
# 基本同步命令格式
sync start -ldir "<本地目录>" -pdir "<云盘目录>" -mode "<模式>"

# 上传本地文件到云盘
sync start -ldir "/home/user/documents" -pdir "/备份盘/文档" -mode "upload"

# 下载云盘文件到本地
sync start -ldir "/home/user/backup" -pdir "/重要文件" -mode "download"
```

### 高级同步选项

```bash
# 指定备份策略
sync start -ldir "/local" -pdir "/cloud" -mode "upload" --policy exclusive

# 指定备份周期
sync start -ldir "/local" -pdir "/cloud" -mode "upload" --cycle infinity

# 指定下载/上传并发数
sync start -ldir "/local" -pdir "/cloud" -mode "upload" --dp 5 --up 3

# 设置分片大小（字节）
sync start -ldir "/local" -pdir "/cloud" -mode "upload" --dbs 1048576 --ubs 1048576
```

**高级选项说明**：
- `--policy`: 备份策略（exclusive 或 increment）
- `--cycle`: 备份周期（infinity 持续或 onetime 一次性）
- `--dp`: 下载并发数
- `--up`: 上传并发数
- `--dbs`: 下载分片大小
- `--ubs`: 上传分片大小
- `--log`: 是否显示日志
- `--ldt`: 本地文件修改检测延迟（秒）
- `--sit`: 扫描文件间隔时间（秒）

### 首次同步优化

首次同步大量文件时，建议先扫描建立数据库：

```bash
# 第一步：扫描建立数据库
sync start -ldir "/local/path" -pdir "/cloud/path" -mode "upload" -step scan

# 第二步：正常启动同步
sync start -ldir "/local/path" -pdir "/cloud/path" -mode "upload"
```

这样可以避免重复同步，提高效率。

### 同步示例

```bash
# 备份文档到云盘（增量备份）
sync start -ldir "/home/user/documents" -pdir "/备份盘/我的文档" -mode "upload" --policy increment

# 备份照片到本地（排他备份）
sync start -ldir "/home/user/photos" -pdir "/相册/重要照片" -mode "download" --policy exclusive

# 持续同步工作目录
sync start -ldir "/home/user/workspace" -pdir "/工作文件" -mode "upload" --cycle infinity
```

**同步注意事项**：
- 支持 JavaScript 插件对备份文件进行过滤
- 同步功能当前为 Beta 版本
- 首次同步大量文件建议先扫描

## 配置管理

### 查看配置

```bash
# 查看当前所有配置
config

# 查看可配置项
config -h
config set -h
```

### 常用配置项

```bash
# 设置下载保存目录
config set -savedir /path/to/download

# 设置下载最大并发数
config set -max_download_parallel 15

# 设置上传最大并发数
config set -max_upload_parallel 5

# 设置下载缓存大小
config set -cache_size 32768

# 限制最大下载速度（字节/秒，0 表示不限制）
config set -max_download_rate 10485760

# 限制最大上传速度（字节/秒，0 表示不限制）
config set -max_upload_rate 5242880

# 设置代理
config set -proxy http://proxy.example.com:8080

# 设置本地网卡地址
config set -local_addrs 192.168.1.100

# 设置域名解析 IP 优先类型
config set -ip_type IPv4

# 组合设置
config set -max_download_parallel 15 -savedir /home/user/downloads
```

### 配置项详细说明

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `-savedir` | 路径 | 下载文件保存目录 |
| `-max_download_parallel` | 整数 | 下载最大并发数（1-15） |
| `-max_upload_parallel` | 整数 | 上传最大并发数（1-10） |
| `-cache_size` | 整数 | 下载缓存大小（KB） |
| `-max_download_rate` | 整数 | 最大下载速度（字节/秒，0=不限制） |
| `-max_upload_rate` | 整数 | 最大上传速度（字节/秒，0=不限制） |
| `-proxy` | URL | 代理服务器地址 |
| `-local_addrs` | IP | 本地网卡地址 |
| `-ip_type` | IPv4/IPv6 | 域名解析 IP 优先类型 |
| `-file_record_config` | 配置 | 文件操作记录功能 |
| `-device_id` | 字符串 | 客户端 ID |

**并发设置建议**：
- 阿里云盘官方 API 对并发有限制
- 如遇下载速度不稳定，建议设置：`config set -max_download_parallel 1`
- 可根据网络情况调整并发数

## 账号管理

### 账号操作

```bash
# 列出所有已登录账号
loglist

# 查看当前账号信息
who

# 切换到指定账号（使用 UID）
su <uid>

# 退出当前账号
logout
```

### 网盘切换

阿里云盘支持多种网盘类型（备份盘、资源库等）：

```bash
# 切换到指定网盘
drive <driveId>

# 查看存储空间使用情况
quota
```

### 登录设备限制

阿里云盘账号有最大登录设备数限制（通常为 10 台）：
- 超出限制时需要在 APP 或 Web 端下线其他设备
- 下线设备后需重启 aliyunpan 程序

## 共享相册

aliyunpan 支持操作阿里云盘的共享相册功能：

### 相册命令

```bash
# 列出所有共享相册
album list
abm list              # 简写

# 列出相册中的文件
album list-file <相册ID>
abm list-file <相册ID>

# 下载相册中的文件
album download-file <相册ID> <文件ID>
abm download-file <相册ID> <文件ID>
```

### 使用示例

```bash
# 查看所有相册
album list

# 查看特定相册的文件
album list-file abc123

# 下载相册中的文件
album download-file abc123 file456
```

**注意**：
- 相册 ID 和文件 ID 可以通过 `album list` 和 `album list-file` 命令获取
- 下载的文件保存到配置的下载目录

## 本地命令

在交互模式中，可以使用本地文件系统命令：

```bash
# 切换本地工作目录
lcd <本地路径>

# 列出本地目录
lls

# 输出本地工作目录
lpwd
```

这些命令用于在交互模式中操作本地文件系统，方便上传时选择文件。

## 工具箱

### 实用工具

```bash
# 显示命令历史
history
history -n 10        # 显示最近 10 条命令
history -n 0         # 显示所有命令历史

# 工具箱
tool

# 显示程序环境变量
env

# 执行系统命令
run <系统命令>

# 检测程序更新
update

# 显示帮助信息
help

# 清空控制台
clear
cls
```

## 常见问题

### 1. 下载速度慢或不稳定

**解决方案**：
```bash
config set -max_download_parallel 1
```

**原因**：
- 需要开通"三方应用权益包"才能享受加速下载
- 阿里云盘官方 API 对并发有限制

### 2. 登录失败或网络连接问题

**最常见原因：代理环境变量冲突**

如果遇到登录失败、网络超时、连接错误等问题，首先检查并清除代理环境变量：

```bash
# 清除所有代理环境变量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# 然后重新启动 aliyunpan
aliyunpan
```

**其他检查事项**：
- 确保完成两次登录流程（API 授权 + 扫码）
- 检查网络连接是否正常
- 查看是否超出设备数限制（最多 10 台）
- 确认防火墙没有阻止 aliyunpan

**注意**：代理环境变量是导致连接问题的最主要原因，务必在使用前清除！

### 3. 同步任务重复文件

**解决方案**：
首次同步前先运行扫描步骤：
```bash
sync start -ldir "/local" -pdir "/cloud" -mode "upload" -step scan
```

### 4. 路径错误

**注意事项**：
- 必须使用正斜杠 `/` 而非反斜杠 `\`
- 包含空格的路径需要用引号包围：`"我的文件夹"`

### 5. 命令不识别

**检查事项**：
- 在交互模式中不需要 `aliyunpan` 前缀
- 在命令行模式中需要 `aliyunpan` 前缀

### 6. VIP 会员要求

**重要提示**：
- 要使用程序进行加速下载，需要在阿里云盘中开通"三方应用权益包"
- 否则无法享受加速下载

## 调试和故障排除

### 启用调试日志

```bash
# Linux/macOS
export ALIYUNPAN_VERBOSE=1
aliyunpan

# Windows
set ALIYUNPAN_VERBOSE=1
aliyunpan
```

启用后可以查看详细的调试日志输出，有助于诊断问题。

## 完整命令参考

详细的命令参考请查看 [references/commands.md](references/commands.md)

## 使用示例

### 示例 1：下载文件

```bash
# 启动交互模式
aliyunpan

# 切换到目标目录
cd /docs

# 查看文件列表
ls

# 下载文件
download report.pdf
```

### 示例 2：上传文件夹

```bash
# 直接在命令行执行
aliyunpan upload /home/user/photos /相册/2024
```

### 示例 3：同步备份

```bash
# 备份本地文档到云盘
aliyunpan sync start -ldir "/home/user/documents" -pdir "/备份盘/文档备份" -mode "upload"
```

### 示例 4：创建分享链接

```bash
# 在交互模式中创建私密分享
cd /重要文件
share set -mode 1 important.pdf
```

### 示例 5：查看共享相册

```bash
# 列出所有相册
album list

# 查看相册文件
album list-file <相册ID>
```

## 参考资源

- **官方仓库**: https://github.com/tickstep/aliyunpan
- **问题反馈**: https://github.com/tickstep/aliyunpan/issues
- **详细命令参考**: 见 references/commands.md

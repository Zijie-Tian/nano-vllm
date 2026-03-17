# aliyunpan 完整命令参考

> 版本：基于 aliyunpan v0.3.7
> 更新时间：2026-02-08

本文档提供 aliyunpan 所有命令的详细参考信息。

## ⚠️ 重要提示：环境变量配置

**在使用 aliyunpan 之前，必须先清除所有代理相关的环境变量！**

代理环境变量（如 `http_proxy`、`https_proxy` 等）会导致 aliyunpan 出现网络连接问题、登录失败、超时等错误。

**每次使用前执行**：

```bash
# 清除所有代理环境变量
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# 然后启动 aliyunpan
aliyunpan
```

**推荐设置别名**：

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
alias aliyunpan='unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY && /usr/local/bin/aliyunpan'
```

---

## 目录

- [账号管理命令](#账号管理命令)
- [文件浏览命令](#文件浏览命令)
- [文件传输命令](#文件传输命令)
- [文件管理命令](#文件管理命令)
- [文件分享命令](#文件分享命令)
- [同步备份命令](#同步备份命令)
- [配置管理命令](#配置管理命令)
- [本地命令](#本地命令)
- [工具箱命令](#工具箱命令)
- [相册命令](#相册命令)

---

## 账号管理命令

### login

登录阿里云盘账号

**用法**：
```bash
login
```

**说明**：
- 使用基于浏览器的授权方式
- 需要完成两次登录流程：
  1. 官方 API 授权登录
  2. Web 界面扫码登录
- 登录成功后会自动保存凭证

**示例**：
```bash
aliyunpan:/ > login
```

### logout

退出当前账号

**用法**：
```bash
logout
```

**说明**：
- 退出当前登录的账号
- 不会删除已保存的凭证
- 可以使用 `su` 命令重新切换回来

**示例**：
```bash
aliyunpan:/ > logout
```

### loglist

列出所有已登录账号

**用法**：
```bash
loglist
```

**说明**：
- 显示所有已登录账号的列表
- 包含账号 UID、昵称等信息
- 用于查看可切换的账号

**示例**：
```bash
aliyunpan:/ > loglist
```

### who

获取当前账号信息

**用法**：
```bash
who
```

**说明**：
- 显示当前登录账号的详细信息
- 包含用户 ID、昵称、VIP 状态等

**示例**：
```bash
aliyunpan:/ > who
```

### su

切换账号

**用法**：
```bash
su <uid>
```

**参数**：
- `<uid>`: 要切换到的账号 UID（通过 `loglist` 查看）

**说明**：
- 快速切换到其他已登录账号
- 不需要重新输入密码

**示例**：
```bash
aliyunpan:/ > su 123456789
```

### drive

切换网盘

**用法**：
```bash
drive <driveId>
```

**参数**：
- `<driveId>`: 网盘 ID

**说明**：
- 阿里云盘支持多种网盘类型（备份盘、资源库等）
- 使用此命令在不同网盘间切换

**示例**：
```bash
aliyunpan:/ > drive backup
```

### quota

查看存储空间使用情况

**用法**：
```bash
quota
```

**说明**：
- 显示当前网盘的存储空间信息
- 包含总容量、已使用容量、剩余容量

**示例**：
```bash
aliyunpan:/ > quota
```

---

## 文件浏览命令

### pwd

显示当前工作目录

**用法**：
```bash
pwd
```

**说明**：
- 显示云盘中的当前工作目录路径

**示例**：
```bash
aliyunpan:/docs > pwd
/docs
```

### cd

切换工作目录

**用法**：
```bash
cd <目录路径>
```

**参数**：
- `<目录路径>`: 目标目录路径
  - 绝对路径：以 `/` 开头，如 `/docs`
  - 相对路径：相对于当前目录，如 `../photos`
  - `..`: 返回上级目录
  - `/`: 返回根目录

**说明**：
- 路径必须使用正斜杠 `/`
- 包含空格的路径需要用引号包围

**示例**：
```bash
aliyunpan:/ > cd /docs
aliyunpan:/docs > cd ..
aliyunpan:/ > cd "我的文件夹"
```

### ls

列出目录内容

**用法**：
```bash
ls [目录路径]
l [目录路径]    # 别名
```

**参数**：
- `[目录路径]`: 可选，要列出的目录路径（默认为当前目录）

**说明**：
- 显示文件和目录的基本信息
- 不带参数时列出当前目录

**示例**：
```bash
aliyunpan:/ > ls
aliyunpan:/ > ls /docs
aliyunpan:/ > l
```

### ll

列出目录内容（详细模式）

**用法**：
```bash
ll [目录路径]
```

**参数**：
- `[目录路径]`: 可选，要列出的目录路径（默认为当前目录）

**说明**：
- 显示文件和目录的详细信息
- 包含文件大小、修改时间、文件 ID 等

**示例**：
```bash
aliyunpan:/ > ll
aliyunpan:/ > ll /docs
```

### tree

列出目录树形图

**用法**：
```bash
tree [选项] [目录路径]
```

**参数**：
- `[目录路径]`: 可选，要显示树形图的目录路径（默认为当前目录）

**选项**：
- `-fp`: 显示完整路径
- `-fs`: 显示文件大小

**说明**：
- 以树形结构显示目录和文件
- 直观展示目录层级关系

**示例**：
```bash
aliyunpan:/ > tree
aliyunpan:/ > tree /docs
aliyunpan:/ > tree -fp /docs
aliyunpan:/ > tree -fs /photos
```

---

## 文件传输命令

### upload

上传文件或目录到云盘

**用法**：
```bash
upload [选项] <本地路径> [本地路径2...] <云盘目录>
u [选项] <本地路径> [本地路径2...] <云盘目录>    # 简写
```

**参数**：
- `<本地路径>`: 要上传的本地文件或目录路径
- `<云盘目录>`: 云盘目标目录

**选项**：
- `-exn <正则表达式>`: 排除匹配正则表达式的文件
- 其他选项请使用 `upload -h` 查看

**说明**：
- 支持上传单个文件、多个文件或整个目录
- 支持断点续传
- 使用正则表达式排除不需要上传的文件

**示例**：
```bash
# 上传单个文件
aliyunpan:/ > upload /home/user/document.pdf /docs

# 上传多个文件
aliyunpan:/ > upload file1.txt file2.txt file3.txt /docs

# 上传目录
aliyunpan:/ > upload /home/user/photos /相册/2024

# 排除临时文件
aliyunpan:/ > upload -exn ".*\.tmp$" /project /backup

# 简写形式
aliyunpan:/ > u /local/file.txt /backup
```

### download

下载文件或目录到本地

**用法**：
```bash
download [选项] <云盘路径> [云盘路径2...]
d [选项] <云盘路径> [云盘路径2...]    # 简写
```

**参数**：
- `<云盘路径>`: 要下载的云盘文件或目录路径

**选项**：
- `--ow`: 覆盖已存在的文件
- `--skip`: 跳过同名文件
- `-p <数量>`: 指定并行线程数
- `--md`: 多用户联合下载
- 其他选项请使用 `download -h` 查看

**说明**：
- 支持下载单个文件、多个文件或整个目录
- 支持断点续传
- 默认保存到配置的下载目录

**示例**：
```bash
# 下载单个文件
aliyunpan:/ > download /docs/report.pdf

# 下载多个文件
aliyunpan:/ > download file1.txt file2.txt file3.txt

# 下载目录
aliyunpan:/ > download /相册/2024

# 覆盖已存在文件
aliyunpan:/ > download --ow /docs/report.pdf

# 指定并行线程数
aliyunpan:/ > download -p 10 /videos/movie.mp4

# 简写形式
aliyunpan:/ > d /docs/report.pdf
```

---

## 文件管理命令

### mkdir

创建目录

**用法**：
```bash
mkdir <目录名>
```

**参数**：
- `<目录名>`: 要创建的目录名称或路径

**说明**：
- 在云盘中创建新目录
- 支持相对路径和绝对路径

**示例**：
```bash
aliyunpan:/ > mkdir /docs/2024
aliyunpan:/ > mkdir "新建文件夹"
aliyunpan:/docs > mkdir project
```

### rm

删除文件或目录

**用法**：
```bash
rm <路径> [路径2...]
```

**参数**：
- `<路径>`: 要删除的文件或目录路径

**说明**：
- 删除的文件会进入回收站
- 可以在网页端恢复
- 支持删除多个文件或目录

**示例**：
```bash
aliyunpan:/ > rm file.txt
aliyunpan:/ > rm /docs/old_folder
aliyunpan:/ > rm file1.txt file2.txt file3.txt
```

### mv

移动文件或目录

**用法**：
```bash
mv <源路径> <目标路径>
```

**参数**：
- `<源路径>`: 源文件或目录路径
- `<目标路径>`: 目标路径

**说明**：
- 移动文件或目录到新位置
- 也可用于跨目录重命名

**示例**：
```bash
aliyunpan:/ > mv /docs/old.txt /archive/old.txt
aliyunpan:/ > mv /photos/2023 /archive/photos_2023
```

### rename

重命名文件或目录

**用法**：
```bash
rename <旧名称> <新名称>
```

**参数**：
- `<旧名称>`: 旧文件或目录名称
- `<新名称>`: 新文件或目录名称

**说明**：
- 只能在同一目录内重命名
- 跨目录重命名请使用 `mv` 命令

**示例**：
```bash
aliyunpan:/docs > rename old_name.txt new_name.txt
aliyunpan:/docs > rename "旧文件夹" "新文件夹"
```

### cp

复制文件或目录

**用法**：
```bash
cp <源路径> <目标路径>
```

**参数**：
- `<源路径>`: 源文件或目录路径
- `<目标路径>`: 目标路径

**说明**：
- 复制文件或目录到新位置

**示例**：
```bash
aliyunpan:/ > cp /docs/file.txt /backup/file.txt
aliyunpan:/ > cp /photos/2024 /backup/photos_2024
```

---

## 文件分享命令

### share set

创建分享链接

**用法**：
```bash
share set [选项] <文件或目录路径>
share s [选项] <文件或目录路径>    # 简写
```

**参数**：
- `<文件或目录路径>`: 要分享的文件或目录路径

**选项**：
- `-mode <模式>`: 设置分享模式
  - `1`: 私密分享（需要提取码）
  - `2`: 公开分享（无需提取码）
- 其他选项请使用 `share -h` 查看

**说明**：
- 创建文件或目录的分享链接
- 生成的链接可以分享给他人
- 模式 1 会生成提取码，更安全
- 模式 2 无需提取码，更方便但安全性较低

**示例**：
```bash
# 默认私密分享
aliyunpan:/ > share set /docs/report.pdf

# 私密分享（需提取码）
aliyunpan:/ > share set -mode 1 /videos/movie.mp4

# 公开分享（无需提取码）
aliyunpan:/ > share set -mode 2 /photos/vacation

# 简写形式
aliyunpan:/ > share s IMG_0106.JPG
```

---

## 同步备份命令

### sync start

启动同步备份任务

**用法**：
```bash
sync start -ldir "<本地目录>" -pdir "<云盘目录>" -mode "<模式>" [选项]
```

**参数**：
- `-ldir <本地目录>`: 本地目录路径
- `-pdir <云盘目录>`: 云盘目录路径
- `-mode <模式>`: 同步模式
  - `upload`: 上传本地文件到云盘
  - `download`: 下载云盘文件到本地
  - `sync`: 双向同步

**选项**：
- `-step <步骤>`: 执行步骤（scan 或正常）
- `--policy <策略>`: 同步策略（exclusive 或 increment）
- `--cycle <周期>`: 同步周期（infinity 或 onetime）
- `--dp <数量>`: 下载并发数
- `--up <数量>`: 上传并发数
- `--dbs <大小>`: 下载分片大小（字节）
- `--ubs <大小>`: 上传分片大小（字节）
- `--log`: 是否显示日志
- `--ldt <秒数>`: 本地文件修改检测延迟（秒）
- `--sit <秒数>`: 扫描文件间隔时间（秒）
- 其他选项请使用 `sync -h` 查看

**说明**：
- 启动同步备份任务
- 支持三种同步模式
- 首次同步建议先扫描建立数据库

**示例**：
```bash
# 上传本地文件到云盘
aliyunpan:/ > sync start -ldir "/home/user/documents" -pdir "/备份盘/文档" -mode "upload"

# 下载云盘文件到本地
aliyunpan:/ > sync start -ldir "/home/user/backup" -pdir "/重要文件" -mode "download"

# 双向同步
aliyunpan:/ > sync start -ldir "/home/user/sync" -pdir "/同步文件夹" -mode "sync"

# 首次同步先扫描
aliyunpan:/ > sync start -ldir "/local" -pdir "/cloud" -mode "upload" -step scan

# 高级参数示例
aliyunpan:/ > sync start -ldir "/local" -pdir "/cloud" -mode "sync" -policy "overwrite" -cycle 3600
```

**首次同步优化**：

首次同步大量文件时，建议分两步执行：

```bash
# 第一步：扫描建立数据库
sync start -ldir "/local/path" -pdir "/cloud/path" -mode "upload" -step scan

# 第二步：正常启动同步
sync start -ldir "/local/path" -pdir "/cloud/path" -mode "upload"
```

---

## 配置管理命令

### config

查看配置

**用法**：
```bash
config
```

**说明**：
- 显示当前所有配置项
- 包含下载目录、并发数等设置

**示例**：
```bash
aliyunpan:/ > config
```

### config set

设置配置

**用法**：
```bash
config set [选项]
```

**选项**：
- `-savedir <路径>`: 设置下载保存目录
- `-max_download_parallel <数量>`: 设置最大下载并发数
- `-max_upload_parallel <数量>`: 设置最大上传并发数
- `-cache_size <大小>`: 设置缓存大小
- `-max_download_rate <速率>`: 设置最大下载速率限制
- `-max_upload_rate <速率>`: 设置最大上传速率限制
- `-proxy <代理地址>`: 设置代理服务器
- `-local_addrs <地址>`: 设置本地网络地址
- `-ip_type <类型>`: 设置 IP 类型（IPv4/IPv6）
- `-file_record_config <配置>`: 设置文件记录配置
- `-device_id <ID>`: 设置设备 ID
- 其他选项请使用 `config set -h` 查看

**说明**：
- 修改程序配置
- 可以同时设置多个配置项

**示例**：
```bash
# 设置下载保存目录
aliyunpan:/ > config set -savedir /path/to/download

# 设置下载最大并发数
aliyunpan:/ > config set -max_download_parallel 10

# 设置上传最大并发数
aliyunpan:/ > config set -max_upload_parallel 5

# 设置代理
aliyunpan:/ > config set -proxy http://127.0.0.1:7890

# 设置下载速率限制（单位：KB/s）
aliyunpan:/ > config set -max_download_rate 10240

# 设置缓存大小
aliyunpan:/ > config set -cache_size 32768

# 组合设置
aliyunpan:/ > config set -max_download_parallel 15 -savedir /home/user/downloads
```

---

## 本地命令

### lcd

切换本地工作目录

**用法**：
```bash
lcd <本地路径>
```

**参数**：
- `<本地路径>`: 本地目录路径

**说明**：
- 在交互模式中切换本地工作目录
- 方便选择上传文件

**示例**：
```bash
aliyunpan:/ > lcd /home/user/documents
```

### lls

列出本地目录

**用法**：
```bash
lls [本地路径]
```

**参数**：
- `[本地路径]`: 可选，本地目录路径（默认为当前本地工作目录）

**说明**：
- 列出本地目录的文件和子目录

**示例**：
```bash
aliyunpan:/ > lls
aliyunpan:/ > lls /home/user
```

### lpwd

输出本地工作目录

**用法**：
```bash
lpwd
```

**说明**：
- 显示当前本地工作目录路径

**示例**：
```bash
aliyunpan:/ > lpwd
/home/user/documents
```

---

## 工具箱命令

### history

显示命令历史

**用法**：
```bash
history [-n <数量>]
```

**选项**：
- `-n <数量>`: 显示最近的命令数量（0 表示所有）

**说明**：
- 显示历史执行过的命令
- 默认显示所有命令

**示例**：
```bash
# 显示所有命令历史
aliyunpan:/ > history

# 显示最近 10 条命令
aliyunpan:/ > history -n 10

# 显示所有命令
aliyunpan:/ > history -n 0
```

### tool

工具箱

**用法**：
```bash
tool
```

**说明**：
- 访问工具箱功能

**示例**：
```bash
aliyunpan:/ > tool
```

### env

显示程序环境变量

**用法**：
```bash
env
```

**说明**：
- 显示程序的环境变量信息

**示例**：
```bash
aliyunpan:/ > env
```

### run

执行系统命令

**用法**：
```bash
run <系统命令>
```

**参数**：
- `<系统命令>`: 要执行的系统命令

**说明**：
- 在交互模式中执行系统命令
- 无需退出 aliyunpan 程序

**示例**：
```bash
aliyunpan:/ > run ls -la
aliyunpan:/ > run cat /etc/hosts
```

### update

检测程序更新

**用法**：
```bash
update
```

**说明**：
- 检查 aliyunpan 程序是否有新版本
- 提示更新信息

**示例**：
```bash
aliyunpan:/ > update
```

### clear / cls

清空控制台

**用法**：
```bash
clear
cls    # 别名
```

**说明**：
- 清空控制台屏幕

**示例**：
```bash
aliyunpan:/ > clear
aliyunpan:/ > cls
```

### help

显示帮助信息

**用法**：
```bash
help [命令名]
```

**参数**：
- `[命令名]`: 可选，显示特定命令的帮助信息

**说明**：
- 显示 aliyunpan 的帮助信息
- 列出所有可用命令
- 可查看特定命令的详细帮助

**示例**：
```bash
# 显示所有命令帮助
aliyunpan:/ > help

# 显示特定命令帮助
aliyunpan:/ > help upload
aliyunpan:/ > help sync
```

---

## 相册命令

### album / abm

共享相册操作

**用法**：
```bash
album <子命令> [选项]
abm <子命令> [选项]    # 简写
```

**子命令**：

#### list
列出所有共享相册

```bash
album list
abm list
```

**说明**：
- 显示所有共享相册列表
- 包含相册 ID、名称、文件数量等信息

**示例**：
```bash
aliyunpan:/ > album list
aliyunpan:/ > abm list
```

#### list-file
列出相册中的文件

```bash
album list-file <相册ID>
abm list-file <相册ID>
```

**参数**：
- `<相册ID>`: 相册的 ID

**说明**：
- 显示指定相册中的所有文件
- 包含文件名、大小、上传时间等信息

**示例**：
```bash
aliyunpan:/ > album list-file 123456
aliyunpan:/ > abm list-file 123456
```

#### download-file
下载相册中的文件

```bash
album download-file <相册ID> [选项]
abm download-file <相册ID> [选项]
```

**参数**：
- `<相册ID>`: 相册的 ID

**选项**：
- `--ow`: 覆盖已存在的文件
- `--skip`: 跳过同名文件
- 其他选项请使用 `album download-file -h` 查看

**说明**：
- 下载指定相册中的文件到本地
- 支持断点续传

**示例**：
```bash
# 下载相册文件
aliyunpan:/ > album download-file 123456

# 覆盖已存在文件
aliyunpan:/ > album download-file 123456 --ow

# 简写形式
aliyunpan:/ > abm download-file 123456
```

**总体说明**：
- 管理共享相册功能
- 具体子命令请使用 `album -h` 查看

---

## 命令别名总结

为了提高使用效率，aliyunpan 提供了许多命令别名：

| 完整命令 | 别名 | 说明 |
|---------|------|------|
| `upload` | `u` | 上传 |
| `download` | `d` | 下载 |
| `ls` | `l` | 列出目录 |
| `ll` | - | 列出目录（详细） |
| `share set` | `share s` | 创建分享 |
| `album` | `abm` | 相册操作 |
| `clear` | `cls` | 清屏 |

---

## 环境变量

### ALIYUNPAN_CONFIG_DIR

设置配置文件目录

**用法**：
```bash
# Linux/macOS
export ALIYUNPAN_CONFIG_DIR=/your/config/path

# Windows
set ALIYUNPAN_CONFIG_DIR=C:\your\config\path
```

**说明**：
- 自定义 aliyunpan 配置文件的存储位置
- 默认使用 XDG 目录规范

### ALIYUNPAN_VERBOSE

启用调试日志

**用法**：
```bash
# Linux/macOS
export ALIYUNPAN_VERBOSE=1

# Windows
set ALIYUNPAN_VERBOSE=1
```

**说明**：
- 启用详细的调试日志输出
- 用于诊断问题

---

## 使用技巧

### 1. Tab 键自动补全

在交互模式中，可以使用 Tab 键自动补全文件名和命令。

### 2. 命令组合

可以在命令行模式下使用 `&&` 连接多个命令：

```bash
aliyunpan cd /docs && aliyunpan ls
```

### 3. 批量操作

许多命令支持批量操作，可以一次处理多个文件：

```bash
# 批量下载
download file1.txt file2.txt file3.txt

# 批量删除
rm file1.txt file2.txt file3.txt
```

### 4. 使用正则表达式

上传时可以使用正则表达式排除不需要的文件：

```bash
# 排除所有 .tmp 和 .log 文件
upload -exn ".*\.(tmp|log)$" /project /backup
```

### 5. 交互模式 vs 命令行模式

- **交互模式**：适合连续操作，无需每次输入 `aliyunpan`
- **命令行模式**：适合脚本化、自动化操作

### 6. 路径技巧

- 使用 `..` 返回上级目录
- 使用 `/` 返回根目录
- 包含空格的路径用引号包围
- 使用绝对路径避免混淆

---

## 注意事项

### ⚠️ 0. 代理环境变量（最重要！）

**使用 aliyunpan 前必须清除所有代理环境变量！**

这是导致连接失败、登录错误、网络超时的最主要原因：

```bash
# 每次使用前执行
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
```

**症状**：
- 登录失败或超时
- 网络连接错误
- 上传/下载失败
- API 调用失败

**解决方案**：
- 清除所有代理环境变量
- 使用别名自动清除：`alias aliyunpan='unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY && /usr/local/bin/aliyunpan'`
- 如需代理，使用 aliyunpan 内置的 `-proxy` 配置

### 1. 路径格式

- 正确：`/docs/file.txt`
- 错误：`\docs\file.txt`

### 2. 引号使用

- 正确：`cd "我的文件夹"`
- 错误：`cd 我的文件夹`

### 3. 并发设置

- 速度不稳定时设置并发数为 1
- 根据网络情况调整

### 4. 设备限制

- 账号最多 10 台设备同时登录
- 超出限制需在 APP 或 Web 端下线

### 5. VIP 要求

- 加速下载需开通"三方应用权益包"

### 6. 已移除的命令

以下命令在 v0.3.7 版本中已不可用：

- `share mc`（秒传链接）
- `import`（导入秒传链接）
- `rapidupload`（导入秒传链接别名）
- `xcp`（跨网盘转存）

如需使用这些功能，请使用其他替代方案或旧版本。

---

## 参考资源

- **官方仓库**: https://github.com/tickstep/aliyunpan
- **问题反馈**: https://github.com/tickstep/aliyunpan/issues
- **更新地址**: https://github.com/tickstep/aliyunpan/releases

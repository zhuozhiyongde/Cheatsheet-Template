#!/bin/bash

# 定义要监控的目录和要执行的脚本
SOURCE_DIR="./source"
COMPILE_SCRIPT="./compile.sh"
INTERVAL=1 # 循环间隔，单位为秒

# 检查目录和脚本是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 监控目录 '$SOURCE_DIR' 不存在。"
    exit 1
fi

if [ ! -f "$COMPILE_SCRIPT" ]; then
    echo "错误: 编译脚本 '$COMPILE_SCRIPT' 不存在。"
    exit 1
fi

if [ ! -x "$COMPILE_SCRIPT" ]; then
    echo "错误: 编译脚本 '$COMPILE_SCRIPT' 没有执行权限。"
    echo "请运行: chmod +x $COMPILE_SCRIPT"
    exit 1
fi

echo "开始监控目录: $SOURCE_DIR"
echo "监控间隔: $INTERVAL 秒"
echo "按 Ctrl+C 停止脚本。"

# 无限循环
while true; do
  # 获取当前时间戳（自纪元以来的秒数）
  now_ts=$(date +%s)
  
  # 查找在过去 $INTERVAL 秒内修改过的 .md 文件
  # -mtime -1s 是 GNU find 的语法，为了兼容性，我们使用一个时间戳文件
  # 创建一个时间戳参考文件，其修改时间是 $INTERVAL 秒之前
  # touch -d @... 是 GNU date 的语法，macOS/BSD 不支持
  # 我们用更兼容的方法：
  touch_ts=$((now_ts - INTERVAL))
  # perl 是一个非常通用的依赖，比 coreutils (gdate) 更常见
  TIMESTAMP_FILE=$(mktemp)
  perl -e "utime $touch_ts, $touch_ts, '$TIMESTAMP_FILE'"

  # 查找比时间戳文件更新的 .md 文件
  # find ... | read 表示如果 find 找到了任何文件，就执行 then 后面的代码块
  if find "$SOURCE_DIR" -type f -name "*.md" -newer "$TIMESTAMP_FILE" | read; then

    echo "----------------------------------------"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - 检测到 .md 文件修改！"
    echo "正在执行编译脚本: $COMPILE_SCRIPT"

    # 执行编译脚本并等待其完成
    /bin/bash "$COMPILE_SCRIPT"

    # 检查上一个命令（即编译脚本）的退出状态
    if [ $? -eq 0 ]; then
        echo "编译成功。"
    else
        echo "警告: 编译脚本执行失败 (退出状态码: $?)"
    fi
    echo "继续监控..."
    echo "----------------------------------------"
  fi
  
  # 清理临时时间戳文件
  rm -f "$TIMESTAMP_FILE"

  # 等待 5 秒
  sleep "$INTERVAL"
done
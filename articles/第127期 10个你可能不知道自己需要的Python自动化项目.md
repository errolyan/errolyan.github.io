# 第127期 10个你可能不知道自己需要的Python自动化项目

![](https://fastly.jsdelivr.net/gh/bucketio/img7@main/2025/10/05/1759634101828-10720018-f659-4ee5-be15-d8b38359e6be.png)


如果你已经有一定的Python使用经验，或许已经掌握了循环、包管理等基础操作，现在想尝试一些超出简单练习范畴的实战项目，那么这里有几个不错的想法。这些项目简单直接、实用性强，且不会花费过多时间。

大多数项目一个下午左右就能完成。下面我们就来具体看看这些项目。

以下是10个小型项目，它们有趣、实用，还能让你小露一手。每个项目都适合在周末完成（大多数不到1小时就能搞定）。

准备好了吗？让我们把Python变成你的得力助手吧。


## 1. 读取银行邮件的个人财务追踪器
为什么需要它？因为手动记录支出只会让人身心俱疲。这个脚本可以登录你的Gmail邮箱，抓取未读的交易邮件，并将交易信息记录到Google表格中。

```python
import imaplib, email, re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

mail = imaplib.IMAP4_SSL("imap.gmail.com")
mail.login("your@gmail.com", "yourpassword")  # 此处替换为你的邮箱和密码
mail.select("inbox")  # 选择收件箱

_, data = mail.search(None, '(UNSEEN SUBJECT "transaction")')  # 搜索未读且主题含"transaction"的邮件

# 配置Google表格连接
scope = ["https://spreadsheets.google.com/feeds"]
creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)  # 需提前准备密钥文件
client = gspread.authorize(creds)
sheet = client.open("Expenses").sheet1  # 打开名为"Expenses"的表格

# 处理每封符合条件的邮件
for num in data[0].split():
    _, msg_data = mail.fetch(num, '(RFC822)')
    msg = email.message_from_bytes(msg_data[0][1])
    body = msg.get_payload(decode=True).decode()  # 提取邮件正文

    # 正则匹配印度卢比（INR）金额
    match = re.search(r'INR\s?([\d,]+\.?\d*)', body)
    if match:
        amount = match.group(1)
        # 将邮件日期、金额、正文前50字符写入表格
        sheet.append_row([msg["date"], amount, body[:50]])
```


## 2. 带语音笔记的每日心情与 productivity 记录器
这个项目能让记日记变得毫不费力。你无需打字，只需说出自己的想法，应用就会自动记录你的心情、工作效率和笔记。快速的语音输入、心情追踪和每日总结功能，能帮你轻松养成记日记的习惯，无需花费时间逐字书写。

```python
import speech_recognition as sr
import pandas as pd
from textblob import TextBlob  # 用于情感分析
from datetime import datetime

def log_mood():
    r = sr.Recognizer()  # 初始化语音识别器
    with sr.Microphone() as source:
        print("说说你的心情吧...")
        audio = r.listen(source)  # 监听麦克风输入
    
    # 将语音转换为文本
    text = r.recognize_google(audio)
    # 分析文本情感极性（数值范围：-1到1，负值表示负面，正值表示正面）
    polarity = TextBlob(text).sentiment.polarity
    # 创建DataFrame存储记录
    df = pd.DataFrame([[datetime.now(), text, polarity]], 
                      columns=["时间", "记录内容", "情感极性"])
    # 将记录追加到CSV文件（无表头，不保留索引）
    df.to_csv("mood_log.csv", mode="a", header=False, index=False)
    print("已记录：", text)

# 调用函数执行记录
log_mood()
```
每周还可以生成简单的图表，查看自己的心情变化趋势。


## 3. Wi-Fi自动修复器
谁也不想凌晨2点还要手动重启路由器吧？这可不该成为一种“习惯”。

```python
import subprocess, time, requests  # subprocess用于执行系统命令，requests用于检测网络连接

def internet_ok():
    try:
        # 尝试访问Google，超时时间设为5秒
        requests.get("https://www.google.com", timeout=5)
        return True  # 能访问则表示网络正常
    except:
        return False  # 访问失败则表示网络异常

# 循环检测网络状态
while True:
    if not internet_ok():
        print("Wi-Fi已断开，正在重启...")
        # 关闭Wi-Fi（"en0"为Wi-Fi适配器名称，需根据实际情况替换）
        subprocess.run(["networksetup", "-setairportpower", "en0", "off"])
        time.sleep(3)  # 等待3秒
        # 重新开启Wi-Fi
        subprocess.run(["networksetup", "-setairportpower", "en0", "on"])
    time.sleep(60)  # 每60秒检测一次网络状态
```
（请将代码中的“en0”替换为你的Wi-Fi适配器名称）。


## 4. 桌面自动清理器
谁的桌面还没有137个名为“截图(92).png”的文件呢？这可算不上什么文件管理方式。

```python
import os, shutil, datetime  # os用于文件操作，shutil用于移动文件

# 获取当前用户的桌面路径
desktop = os.path.expanduser("~/Desktop")
# 遍历桌面上的所有文件
for file in os.listdir(desktop):
    path = os.path.join(desktop, file)  # 拼接完整文件路径
    if os.path.isfile(path):  # 仅处理文件（不处理文件夹）
        # 获取文件扩展名并转为小写
        ext = os.path.splitext(file)[1].lower()
        # 根据扩展名分类：图片（.png/.jpg）或文档（其他格式）
        category = "Images" if ext in [".png", ".jpg"] else "Docs"
        # 拼接分类文件夹路径
        folder = os.path.join(desktop, category)
        # 若文件夹不存在则创建（exist_ok=True避免报错）
        os.makedirs(folder, exist_ok=True)
        # 将文件移动到对应分类文件夹
        shutil.move(path, os.path.join(folder, file))
```
可以通过定时任务调度工具（如cron、任务计划程序）设置每日自动运行。


## 5. YouTube播放列表下载器+摘要生成器
一个3小时的教程，其实15分钟就能掌握核心内容——这个项目就能帮你实现。

```python
from pytube import YouTube, Playlist  # pytube用于下载YouTube视频
import openai  # OpenAI用于后续生成摘要（需配置API密钥）

# 初始化播放列表（替换为目标播放列表URL）
playlist = Playlist("https://www.youtube.com/playlist?list=...")
# 下载播放列表中的前2个视频（仅音频）
for url in playlist.video_urls[:2]:
    yt = YouTube(url)
    # 筛选仅音频的流并选择第一个
    stream = yt.streams.filter(only_audio=True).first()
    # 下载音频并命名为"video.mp3"
    stream.download(filename="video.mp3")
    print(f"已下载：{yt.title}，后续可生成摘要...")
```
可添加Whisper（OpenAI的语音转文字工具）或OpenAI API来生成音频转录文本及摘要。


## 6. 智能截图分类器
你肯定不记得那张表情包截图是上周保存的还是去年保存的吧？这个项目能帮你解决这个问题。

```python
import os, pytesseract  # pytesseract用于图片文字识别
from PIL import Image  # PIL用于处理图片
from datetime import datetime

# 获取截图文件夹路径（默认在用户图片目录下的Screenshots文件夹）
folder = os.path.expanduser("~/Pictures/Screenshots")
# 遍历文件夹中的文件
for file in os.listdir(folder):
    if file.endswith(".png"):  # 仅处理PNG格式截图
        # 打开图片
        img = Image.open(os.path.join(folder, file))
        # 识别图片中的文字，取前30个字符并替换换行符为空格
        text = pytesseract.image_to_string(img)[:30].replace("\n", " ")
        # 获取当前日期并格式化为"年月日"（如20250928）
        timestamp = datetime.now().strftime("%Y%m%d")
        # 生成新文件名：日期_识别文字.png
        new_name = f"{timestamp}_{text}.png"
        # 重命名文件
        os.rename(os.path.join(folder, file), 
                  os.path.join(folder, new_name))
```
现在，你的表情包截图可以通过关键词搜索找到了。


## 7. GitHub星标仪表盘
谁没偷偷刷新页面，查看自己的仓库是否多了星标呢？这个项目能帮你直观追踪星标变化。

```python
import requests, matplotlib.pyplot as plt  # requests用于调用API，matplotlib用于绘图

# 目标仓库（格式：用户名/仓库名，此处以octocat的Hello-World为例）
repo = "octocat/Hello-World"
# 构造GitHub星标API请求URL
url = f"https://api.github.com/repos/{repo}/stargazers"
# 发送请求（指定API版本为v3，获取星标详细信息）
r = requests.get(url, headers={"Accept": "application/vnd.github.v3.star+json"})
# 提取每个星标的日期（仅保留"年-月-日"部分）
dates = [s["starred_at"][:10] for s in r.json()]
# 绘制星标日期分布直方图（10个区间）
plt.hist(dates, bins=10)
# 旋转x轴标签，避免重叠（旋转45度）
plt.xticks(rotation=45)
# 设置图表标题
plt.title("星标增长趋势")
# 显示图表
plt.show()
```
下次站会时，就可以展示这个图表啦。


## 8. 冰箱食材食谱查询器
盯着半个西红柿和剩米饭发呆可想不出什么好吃的——这个项目能帮你根据现有食材找到合适的食谱。

```python
import requests  # 用于调用食谱API

# 现有食材（可根据实际情况修改，用英文逗号分隔）
ingredients = "tomato,rice"
# 构造Spoonacular食谱API请求URL（需替换为你的API密钥）
url = f"https://api.spoonacular.com/recipes/findByIngredients?ingredients={ingredients}&number=3&apiKey=YOUR_KEY"
# 发送请求并解析JSON响应
resp = requests.get(url).json()
# 遍历响应结果，打印食谱名称和来源链接
for r in resp:
    print(r["title"], "->", r["sourceUrl"])
```
一瞬间，你也能变身大厨了。


## 9. Slack通知自动静音器
不是每一条“嗨”都值得你立即回复，这个项目能帮你过滤不必要的干扰。

```python
from slack_sdk import WebClient  # slack_sdk用于操作Slack
import time

# 初始化Slack客户端（需替换为你的Slack令牌）
client = WebClient(token="xoxb-your-token")
# 消息数量阈值（超过此数量则静音）
threshold = 10
# 目标Slack频道ID（需替换为实际频道ID）
channel = "C12345678"

# 循环检测频道消息
while True:
    # 获取频道历史消息（最多获取threshold条）
    msgs = client.conversations_history(channel=channel, limit=threshold).data
    # 若消息数量超过阈值，则设置频道静音
    if len(msgs["messages"]) > threshold:
        client.conversations_info(channel=channel)  # 获取频道信息（验证频道有效性）
        # 设置频道用途为“暂时静音”，提示他人当前频道已静音
        client.conversations_setPurpose(channel=channel, purpose="Muted temporarily")
        print("频道已静音！")
        time.sleep(1800)  # 静音30分钟（1800秒）
    time.sleep(300)  # 每5分钟（300秒）检测一次
```
这下，你的内心平静终于可以回来了。


## 10. PDF合同分析器
为什么需要它？因为逐字阅读合同里的小字条款简直让人崩溃。

```python
import pdfplumber, re  # pdfplumber用于提取PDF文本，re用于正则匹配

# 打开PDF合同文件（替换为你的合同文件路径）
with pdfplumber.open("contract.pdf") as pdf:
    text = ""
    # 提取每一页的文本并拼接
    for page in pdf.pages:
        text += page.extract_text()

# 需要重点关注的关键词（如“终止条款”“费用”“不可退款”“责任”）
keywords = ["termination", "fees", "non-refundable", "liability"]
# 遍历关键词，检测合同中是否包含这些内容（不区分大小写）
for word in keywords:
    if re.search(word, text, re.IGNORECASE):
        print(f"⚠️ 已发现关键词：{word}")
```
虽然它不能替代律师，但绝对能成为律师的得力助手。
# Assignment 4 Report

### 12312515 王洛源

本次作业要求基于LLM框架搭建一个部署在社交软件，能直接和用户交流的 Agent。由于在本学期其他课程中，我已经在我本地Windows主机部署好了 openclaw，我本次作业选择了比较容易配置的 Openclaw QQ Bot。主要流程如下：

1. 在QQ开放平台进行注册并认证，创建机器人，获取AppID和AppSecret

2. 在本地终端启动 openclaw gateway ，然后安装官方的qqbot插件

3. 用之前获取的AppID和AppSecret配置通道，然后重启 gateway，在QQ端进行测试

配置通道用的指令如下：

```
openclaw channels add --channel qqbot --token "AppID:AppSecret"
```

测试结果一切正常，该机器人也继承了我之前在网页端和openclaw交流的记忆，不过只有我本地主机的gateway启动的情况下机器人才会有反应，也就是说，在使用这个bot前，我必须先在我自己的电脑上启动终端，然后运行openclaw gateway。

下面是我使用这个bot完成部分任务的截图：

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-05-01-16-37-20-image.png" title="" alt="" width="809">

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-05-01-16-51-30-image.png" title="" alt="" width="809">

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-05-01-17-10-04-image.png" title="" alt="" width="812">

![](C:\Users\lenovo\AppData\Roaming\marktext\images\2026-05-01-17-21-18-image.png)

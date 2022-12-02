### 关于分类
- 分类类别：依次是2 6 12 分类问题，类别如下

```
dict_2class = {0:'Novpn',1:'Vpn'}
dict_6class_novpn = {0:'Chat',1:'Email',2:'File',3:'P2p',4:'Streaming',5:'Voip'}
dict_6class_vpn = {0:'Vpn_Chat',1:'Vpn_Email',2:'Vpn_File',3:'Vpn_P2p',4:'Vpn_Streaming',5:'Vpn_Voip'}
dict_12class = {0:'Chat',1:'Email',2:'File',3:'P2p',4:'Streaming',5:'Voip',6:'Vpn_Chat',7:'Vpn_Email',8:'Vpn_File',9:'Vpn_P2p',10:'Vpn_Streaming',11:'Vpn_Voip'}
```
### 关于数据来源

&emsp;&emsp;这里是直接使用论文作者给出的预处理好的数据，处理工具原始文章的仓库也以及给出

### 使用 
- 运行`1d_cnn/cnn_1d_torch`进行12分类
- 修改29-36行换数据集
- 修改26行的`label_num`变量与38行更换任务

> `1d_cnn/cnn_1d_tensorflow`是原文代码，这里并没有调试，供参考

原文代码：
``https://github.com/mydre/wang-wei-s-research``

博客地址：
``https://blog.csdn.net/qq_45125356/article/details/126956497?spm=1001.2014.3001.5501``


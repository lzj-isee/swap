import requests
import os
def getPages():#只能得到30张图片，想得到更多图片，需要变化params['pn']的值。
    headers={'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.134 Safari/537.36 Edg/103.0.1264.77"}
    url='https://image.baidu.com/search/acjson'
    #追加请求内容（负载）
    params={
        'tn':'resultjson_com',
        'logid':'5336511566462226640',
        'ipn':'rj',
        'ct':'201326592',
        'is':' ',
        'fp':'result',
        'fr':' ',
        'word':'戴帽子的人',
        'cg':'girl',
        'queryWord':'戴帽子的人',
        'cl':'2',
        'lm':'-1',
        'ie':'utf-8',
        'oe':'utf-8',
        'adpicid':' ',
        'st':' ',
        'z':' ',
        'ic':' ',
        'hd':' ',
        'latest':' ',
        'copyright':' ',
        's':' ',
        'se':' ',
        'tab':' ',
        'width':' ',
        'height':' ',
        'face':' ',
        'istype':' ',
        'qc':' ',
        'nc':'1',
        'expermode':' ',
        'nojc':' ',
        'isAsync':' ',
        'pn':'0',
        'rn':'15',
        'gsm':'1e',
        '1660570401395':' '
        }    
    res=requests.get(url=url,headers=headers,params=params)
    # print(res.json())
    data=res.json()['data']
    # print(data)
    #得到所有图片地址
    urlPages=[]
    for i in data:
        if i.get('thumbURL') !=None:          
            urlPages.append(i['thumbURL'])
    # print(urlPages)   
    #检测文件夹是否存在
    dir='./baidu'
    if not os.path.exists(dir):
        os.mkdir(dir)#创建目录方法
    #向每个图片url发起请求
    x=0
    for o in urlPages:
        print('下载成功')
        res=requests.get(url=o,headers=headers)
        #下载到dir文件夹
        open(f'{dir}/{x}.jpg','wb').write(res.content)
        x+=1
getPages()
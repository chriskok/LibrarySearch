import pandas as pd
import urllib
import urllib2

df = pd.read_csv("chipdata.csv", delimiter=',', encoding="utf-8-sig")

camera_id = 10296

url_df = df.loc[df['camera_id'] == camera_id]
url_df2 = url_df['curr_url']

success = 0

# urllib.URLopener.version = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36"
# urllib.urlretrieve('https://i.ytimg.com/vi/2Rf01wfbQLk/maxresdefault.jpg', "test.jpg")

# image = urllib.URLopener()
# image.retrieve('https://faa-crowd-wx.s3.amazonaws.com/10296-151994916000000000.jpg?Signature=iYKaNQ%2FvkOUiCsIDWNYBWCpYdSk%3D&Expires=1567369105&AWSAccessKeyId=AKIASBJCP53P67LK66V5', "test.jpg")
urllib.urlretrieve('https://faa-crowd-wx.s3.amazonaws.com/10296-151994916000000000.jpg?Signature=iYKaNQ%2FvkOUiCsIDWNYBWCpYdSk%3D&Expires=1567369105&AWSAccessKeyId=AKIASBJCP53P67LK66V5', "test")

#
# for row in range(75, len(url_df2)):
#     todownload = url_df2[row][10:-3]
#
#     try:
#         request = urllib2.Request(todownload)
#         result = urllib2.urlopen(request)
#         contents = result.read()
#     except:
#         continue
#
#     print('downloading: ' + todownload)
#     urllib.urlretrieve(todownload, "downloaded_imgs/" + str(row) + ".jpg")
#
#     # f = open("downloaded_imgs/" + str(row) + ".jpg",'wb')
#     # f.write(urllib.urlopen(todownload).read())
#     # f.close()
#
#     success += 1
#     if (success == 10):
#         break

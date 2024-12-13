import sys

from pytubefix import YouTube
from pytubefix.cli import on_progress

url = sys.argv[1]
RES = sys.argv[2]

yt = YouTube(url, on_progress_callback = on_progress)

for idx,i in enumerate(yt.streams):
   if i.resolution == RES:
      print(idx)
      print(i.resolution)
      break

print(yt.streams[idx])
yt.streams[idx].download()

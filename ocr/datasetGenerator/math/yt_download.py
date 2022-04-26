import pytube
link = "https://www.youtube.com/watch?v=uKwD3JuRWeA"
yt = pytube.YouTube(link)
stream = yt.streams.first()
stream.download()
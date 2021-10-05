import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import requests
import json

legal_colors = [
    "#FFFFFF",
    "#E4E4E4",
    "#888888",
    "#222222",
    "#FFA7D1",
    "#CF6EE4",
    "#820080",
    "#E50000",
    "#E5D900",
    "#E59500",
    "#94E044",
    "#02BE01",
    "#A06A42",
    "#00D3DD",
    "#0083C7",
    "#0000EA",
]

cmap = colors.ListedColormap(legal_colors)
bounds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
norm = colors.BoundaryNorm(bounds, cmap.N)

board = np.zeros((50, 50))

response = requests.get("https://nabla.no/place/3/history")
print(response.status_code)


def jprint(obj):
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)


#jprint(response.json())
data = json.loads(response.content.decode('utf-8'))
#print(data['grid'][0][0])

fig, ax = plt.subplots()
im = plt.imshow(board, cmap=cmap, norm=norm)


def animate(i):
    x = data[i]["x"]
    y = data[i]["y"]
    board[y][x] = legal_colors.index("#"+data[i]['color'])
    im.set_data(board)
    return [im]


anim = FuncAnimation(fig, animate, frames=len(data), interval=20, blit=True, save_count=len(data))
anim.save('Nablaplace_timelapse2.mp4')
plt.show()


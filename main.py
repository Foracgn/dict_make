from fontTools.ttLib import TTFont
from fontTools import subset

if __name__ == '__main__':
    font = TTFont("./AYJGW20200206.ttf")
    fontMap = font.getBestCmap()
    print(fontMap)

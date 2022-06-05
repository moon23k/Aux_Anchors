import pandas as pd
from konlpy.tag import Mecab


#우선적으로 news data를 사용하고, 혹시 데이터가 부족하면 추후보강예정
def run():
	news_1 = pd.read_excel('data/nmt/news_1.xlsx')
	news_2 = pd.read_excel('data/nmt/news_2.xlsx')
	news_3 = pd.read_excel('data/nmt/news_3.xlsx')
	news_4 = pd.read_excel('data/nmt/news_4.xlsx')

	src, trg = [], []

	src.extend(news_1.iloc[:, -1].tolist())
	src.extend(news_2.iloc[:, -1].tolist())
	src.extend(news_3.iloc[:, -1].tolist())
	src.extend(news_4.iloc[:, -1].tolist())

	trg.extend(news_1.iloc[:, -2].tolist())
	trg.extend(news_2.iloc[:, -2].tolist())
	trg.extend(news_3.iloc[:, -2].tolist())
	trg.extend(news_4.iloc[:, -2].tolist())


	src_train, src_valid, src_test = src[:-2000], src[-2000:-1000], src[-1000:]
	trg_train, trg_valid, trg_test = trg[:-2000], trg[-2000:-1000], trg[-1000:]



	#pretokenizing with sacremoses for English seq
	




	#pretokenizing with mecab for Korean seq
    trg_train = [mecab.morphs(seq) for seq in trg_train]
    trg_valid = [mecab.morphs(seq) for seq in trg_valid]
    trg_test = [mecab.morphs(seq) for seq in trg_test]

    with open('data/nmt/seq/train.src', 'w') as f:
        f.write('\n'.join(src_train))
    with open('data/nmt/seq/valid.src', 'w') as f:
        f.write('\n'.join(src_valid))
    with open('data/nmt/seq/test.src', 'w') as f:
        f.write('\n'.join(src_test))


    with open('data/nmt/seq/train.trg', 'w') as f:
        f.write('\n'.join(trg_train))
    with open('data/nmt/seq/valid.trg', 'w') as f:
        f.write('\n'.join(trg_valid))
    with open('data/nmt/seq/test.trg', 'w') as f:
        f.write('\n'.join(trg_test))




    #mecab
	mecab = Mecab()
	text = "뉴욕의 펜트하우스에서 원고를 쓰기 시작했지만, 뉴욕의 여름 날씨와 시끄러운 소음에 괴러워 하던 차에, 롱아일랜드로 이사하게 된다."
	print(mecab.morphs(text))


if __name__ == '__main__':
	run()
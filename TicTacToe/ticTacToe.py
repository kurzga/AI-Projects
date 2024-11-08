import numpy as np
import pickle

satir = 3
sutun = 3


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((satir, sutun))
        self.p1 = p1
        self.p2 = p2
        self.bittimi = False
        self.OyunDurumu = None

        #1. oyuncuyu varsayılan olarak ilk başlayan kabul et.
        self.oyuncuSembolu = 1

    
    def OyunuGoster(self):
        self.OyunDurumu = str(self.board.reshape(sutun * satir))
        return self.OyunDurumu

    def kazanan(self):

        # satır toplamı ödül tablosunda 1 veya -1'dir.3 veya -3 olursa oyun biter.
        for i in range(satir):
            if sum(self.board[i, :]) == 3:
                self.bittimi = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.bittimi = True
                return -1
            
        # sütun toplamı da ödül tablosunda 1 veya -1'dir.3 veya -3 olursa oyun biter.
        for i in range(sutun):
            if sum(self.board[:, i]) == 3:
                self.bittimi = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.bittimi = True
                return -1
        # Çapraz kısımların toplamını da burada hesaplıyoruz.
        capraz_toplam1 = sum([self.board[i, i] for i in range(sutun)])
        capraz_toplam2 = sum([self.board[i, sutun - i - 1] for i in range(sutun)])
        capraz_toplam = max(abs(capraz_toplam1), abs(capraz_toplam2))
        if capraz_toplam == 3:
            self.bittimi = True
            if capraz_toplam1 == 3 or capraz_toplam2 == 3:
                return 1
            else:
                return -1
            
        # beraberlik kontrolü burada oluyor.

        
        # Yapılabilecek hamleler var mı?
        if len(self.yapılabilirHamleler()) == 0:
            self.bittimi = True
            return 0
        # bitmediyse bittimi değişkenine false atıyoruz.
        self.bittimi = False  
        return None
    
    def yapılabilirHamleler(self):
        hamleler = []
        for i in range(satir):
            for j in range(sutun):
                if self.board[i, j] == 0:
                    hamleler.append((i, j))
        return hamleler

    def DurumuGuncelle(self, konum):
        self.board[konum] = self.oyuncuSembolu
        #Oyuncuyu değiştirme
        self.oyuncuSembolu = -1 if self.oyuncuSembolu == 1 else 1

    # Eğer oyun bittiyse çağrılacak fonksiyon
    def ÖdülVer(self):
        result = self.kazanan()

        if result == 1:
            self.p1.OyuncuyaOdulEkle(1)
            self.p2.OyuncuyaOdulEkle(0)
        elif result == -1:
            self.p1.OyuncuyaOdulEkle(0)
            self.p2.OyuncuyaOdulEkle(1)
        else:
            self.p1.OyuncuyaOdulEkle(0.1)
            self.p2.OyuncuyaOdulEkle(0.5)

    # board reset
    def reset(self):
        self.board = np.zeros((satir, sutun))
        self.OyunDurumu = None
        self.bittimi = False
        self.oyuncuSembolu = 1

    def oyna(self, rounds=100):
        for i in range(rounds):

            while not self.bittimi:
                # 1. oyuncunun hamleleri
                hamleler = self.yapılabilirHamleler()
                p1_action = self.p1.HamleSec(hamleler, self.board, self.oyuncuSembolu)
                # Hamleyi yap ve tablonun(board) durumunu değiştir.
                self.DurumuGuncelle(p1_action)
                board_hash = self.OyunuGoster()
                self.p1.HamleEkle(board_hash)
                # OyunDurumunu kontrol et(Oyun Bitti mi?)

                win = self.kazanan()
                if win is not None:
                    # Oyun bitmediyse Q table güncellenir,oyuncular sfırlanır.
                    self.ÖdülVer()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Oyuncu 2 kontrolü
                    hamleler = self.yapılabilirHamleler()
                    p2_hamle = self.p2.HamleSec(hamleler, self.board, self.oyuncuSembolu)
                    self.DurumuGuncelle(p2_hamle)
                    OyunDurumu = self.OyunuGoster()
                    self.p2.HamleEkle(OyunDurumu)

                    win = self.kazanan()
                    if win is not None:
                        # Oyun kazan ya da kaybet 2. oyuncu ile bitiyor.O yüzden önce 1. oyuncuyu reset yapıyoruz.
                        self.ÖdülVer()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break

    # oyna_insan fonksiyonu bir önceki fonksiyondan farklı olarak Q learning ile eğitilmiş ajanın insanla oynamasını sağlıyor.
    def oyna_insan(self):
        while not self.bittimi:
            # 1. Oyuncu kontrolü
            hamleler = self.yapılabilirHamleler()
            p1_action = self.p1.HamleSec(hamleler, self.board, self.oyuncuSembolu)
            self.DurumuGuncelle(p1_action)
            self.Goster()
            # Kazanan kontrolü
            win = self.kazanan()
            if win is not None:
                if win == 1:
                    print(self.p1.isim, "kazandi")
                else:
                    print("BERABERLİK")
                self.reset()
                break

            else:
                # 2. Oyuncu kontrolü
                hamleler = self.yapılabilirHamleler()
                p2_action = self.p2.HamleSec(hamleler)

                self.DurumuGuncelle(p2_action)
                self.Goster()
                win = self.kazanan()
                if win is not None:
                    if win == -1:
                        print(self.p2.isim, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def Goster(self):
        # p1: o   p2: x
        for i in range(0, satir):
            print('-------------')
            out = '| '
            for j in range(0, sutun):
                if self.board[i, j] == 1:
                    isaret = 'o'
                if self.board[i, j] == -1:
                    isaret = 'x'
                if self.board[i, j] == 0:
                    isaret = ' '
                out += isaret + ' | '
            print(out)
        print('-------------')


class Player:
    def __init__(self, isim, exp_rate=0.3):
        self.isim = isim
        self.hamleler = []
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.hamleler_value = {}

    def OyunuGoster(self, board):
        OyunDurumu = str(board.reshape(sutun * satir))
        return OyunDurumu

    def HamleSec(self, hamleler, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # rastgele hamle
            index = np.random.choice(len(hamleler))
            action = hamleler[index]
        else:
            value_max = -999
            for p in hamleler:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.OyunuGoster(next_board)
                value = 0 if self.hamleler_value.get(next_boardHash) is None else self.hamleler_value.get(next_boardHash)
               
                if value >= value_max:
                    value_max = value
                    action = p
        
        return action

    def HamleEkle(self, state):
        self.hamleler.append(state)

    
    def OyuncuyaOdulEkle(self, reward):
        for st in reversed(self.hamleler):
            if self.hamleler_value.get(st) is None:
                self.hamleler_value[st] = 0
            self.hamleler_value[st] += self.lr * (self.decay_gamma * reward - self.hamleler_value[st])
            reward = self.hamleler_value[st]

    def reset(self):
        self.hamleler = []

    def savePolicy(self):
        fw = open('policy_' + str(self.isim), 'wb')
        pickle.dump(self.hamleler_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.hamleler_value = pickle.load(fr)
        fr.close()


class HumanPlayer:
    def __init__(self, isim):
        self.isim = isim

    def HamleSec(self, hamleler):
        while True:
            row = int(input("satır seçiniz:"))
            col = int(input("sütun seçiniz:"))
            action = (row, col)
            if action in hamleler:
                return action

    def HamleEkle(self, state):
        pass


    def OyuncuyaOdulEkle(self, reward):
        pass

    def reset(self):
        pass


if __name__ == "__main__":
    # ajan eğitimi
    p1 = Player("p1")
    p2 = Player("p2")
    
    st = State(p1, p2)
    print("eğitiliyor...")
    st.oyna(1)
    p1.savePolicy()
    p2.savePolicy()

    p1 = Player("oyuncu_1", exp_rate=0)
    p1.loadPolicy("policy_p1")

    p2 = HumanPlayer("oyuncu_2")

    while True:
        st = State(p1, p2)
        st.oyna_insan()
        print("Tekrar oynamak istiyor musun?(0[hayır],1[evet]):")
        secim = int(input("secim:"))
        if secim == 0:
            break
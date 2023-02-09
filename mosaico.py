import skimage.io as io
import os 
import numpy as np
import cv2 
import matplotlib.pyplot as plt


def mosaico(list_img,path_save) : 
    """ Questa funzione ti genera un mosaico d' immagini, molto utile per fare un overview su dataset
        vengono organizzate in 5 righe e 2 colonne 

    Args:
        list_img (list): lista delle immagini da fare il mosaico
        path_save (str): path in cui vuoi andare a salvare le immagini 
        
    """
    
        
    def generator(lista):
        check = 0
        for i in range(rows):
            if (i+1)*cols <= len(lista):
                yield [lista[j] for j in range(i*cols, (i+1)*cols)]
            else:
                if (i+1)*cols - len(lista) != cols and check == 0:
                    A =  [lista[j] for j in range((i)*cols,len(lista))]
                    check = 1
                    for i in range(((i+1)*cols - len(lista))):
                        A.append("empty")
                    yield A
                else:
                    yield ['empty' for j in range(i*cols, (i+1)*cols)]

    def generator2(lista):
    # check = 0 
        n_imm = 10
        n_list = round(len(lista)/n_imm)
        for i in range(n_list):
            if (i+1)*n_imm < len(lista): 
                yield[lista[j] for j in range(i*n_imm,(i+1)*n_imm)] 
            else : 
                yield[lista[j] for j in range(i*n_imm,len(lista))]



    gen1 = generator2(list_img)
    for i in range(round(len(list_img)/10)):
        mosaico = []
        raw_mosaico = []
        textes = []
        mosaico_textes = []

    
        listaa = next(gen1)
        rows = 5
        cols =  2
        gen = generator(listaa)


        for name in range(rows):
            lista = next(gen)
    
            for name in lista:
                if name != "empty" :
                    img = io.imread(name)
                    img = cv2.resize(img, (224,224), interpolation =cv2.INTER_AREA)
                    texted = cv2.putText(img= np.zeros((224,224), dtype= np.float32), text=   str(os.path.dirname(name))[:15], org=(20,200),fontFace=1, fontScale=0.4 color=(1.,0 ,0), thickness=0.4)
                    raw_mosaico.append(img)
                    textes.append(texted)
            
                else:
                    raw_mosaico.append(np.ones((224,224)))
                    textes.append(np.ones((224,224)))
            
            mosaico.append(np.concatenate(tuple(raw_mosaico), axis = 1))
            mosaico_textes.append(np.concatenate(tuple(textes), axis = 1))
            raw_mosaico = []
            textes = []

        mosaico = np.concatenate(tuple(mosaico), axis = 0)
        mosaico_textes = np.concatenate(tuple(mosaico_textes), axis = 0)

    
        plt.figure(figsize=(40, 20))
        plt.imshow(mosaico,cmap = "gray")
        plt.imshow(mosaico_textes, cmap = 'Reds', alpha = 0.5)
        plt.savefig(path_save + str(i) + ".tiff" ,dpi = 400)

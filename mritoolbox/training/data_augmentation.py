def matrice_translation(path, pixel_max=10, trans_rate=0.5):

    translation = random.randint(1, 100)

    if translation >= trans_rate*100 :

        which_translation = random.randint(1,14)
        pixel_translation = random.randint(1, pixel_max)
        
        wm_old_matrice = nib.load(path['wm']).get_data()
        wm_matrice = np.zeros(SIZE[0:3])
        
        gm_old_matrice = nib.load(path['gm']).get_data()
        gm_matrice = np.zeros(SIZE[0:3])

        csf_old_matrice = nib.load(path['csf']).get_data()
        csf_matrice = np.zeros(SIZE[0:3])

        raw_old_matrice = nib.load(path['raw_proc']).get_data()
        raw_matrice = np.zeros(SIZE[0:3])

        if which_translation == 1:
            wm_matrice [pixel_translation:, :, :] = wm_old_matrice[pixel_translation:, :, :] 
            gm_matrice [pixel_translation:, :, :] = gm_old_matrice[pixel_translation:, :, :] 
            csf_matrice [pixel_translation:, :, :] = csf_old_matrice[pixel_translation:, :, :] 
            raw_matrice [pixel_translation:, :, :] = raw_old_matrice[pixel_translation:, :, :] 
        if which_translation == 2:
            wm_matrice [pixel_translation:, pixel_translation:, :] = wm_old_matrice[pixel_translation:, pixel_translation:, :]
            gm_matrice [pixel_translation:, pixel_translation:, :] = gm_old_matrice[pixel_translation:, pixel_translation:, :]
            csf_matrice [pixel_translation:, pixel_translation:, :] = csf_old_matrice[pixel_translation:, pixel_translation:, :]
            raw_matrice [pixel_translation:, pixel_translation:, :] = raw_old_matrice[pixel_translation:, pixel_translation:, :]
        if which_translation == 3:
            wm_matrice [pixel_translation:, :, pixel_translation:] = wm_old_matrice[pixel_translation:, :, pixel_translation:]
            gm_matrice [pixel_translation:, :, pixel_translation:] = gm_old_matrice[pixel_translation:, :, pixel_translation:]
            csf_matrice [pixel_translation:, :, pixel_translation:] = csf_old_matrice[pixel_translation:, :, pixel_translation:]
            raw_matrice [pixel_translation:, :, pixel_translation:] = raw_old_matrice[pixel_translation:, :, pixel_translation:]
        if which_translation == 4:
            wm_matrice [pixel_translation:, pixel_translation:, pixel_translation:] = wm_old_matrice[pixel_translation:, pixel_translation:, pixel_translation:]
            gm_matrice [pixel_translation:, pixel_translation:, pixel_translation:] = gm_old_matrice[pixel_translation:, pixel_translation:, pixel_translation:]
            csf_matrice [pixel_translation:, pixel_translation:, pixel_translation:] = csf_old_matrice[pixel_translation:, pixel_translation:, pixel_translation:]
            raw_matrice [pixel_translation:, pixel_translation:, pixel_translation:] = raw_old_matrice[pixel_translation:, pixel_translation:, pixel_translation:]
        if which_translation == 5:
            wm_matrice [:, pixel_translation:, :] = wm_old_matrice[:, pixel_translation:, :]
            gm_matrice [:, pixel_translation:, :] = gm_old_matrice[:, pixel_translation:, :]
            csf_matrice [:, pixel_translation:, :] = csf_old_matrice[:, pixel_translation:, :]
            raw_matrice [:, pixel_translation:, :] = raw_old_matrice[:, pixel_translation:, :]
        if which_translation == 6:
            wm_matrice [:, pixel_translation:, pixel_translation:] = wm_old_matrice[:, pixel_translation:, pixel_translation:]
            gm_matrice [:, pixel_translation:, pixel_translation:] = gm_old_matrice[:, pixel_translation:, pixel_translation:]
            csf_matrice [:, pixel_translation:, pixel_translation:] = csf_old_matrice[:, pixel_translation:, pixel_translation:]
            raw_matrice [:, pixel_translation:, pixel_translation:] = raw_old_matrice[:, pixel_translation:, pixel_translation:]
        if which_translation == 7:
            wm_matrice [:, :, pixel_translation:] = wm_old_matrice[:, :, pixel_translation:] 
            gm_matrice [:, :, pixel_translation:] = gm_old_matrice[:, :, pixel_translation:] 
            csf_matrice [:, :, pixel_translation:] = csf_old_matrice[:, :, pixel_translation:] 
            raw_matrice [:, :, pixel_translation:] = raw_old_matrice[:, :, pixel_translation:] 
        if which_translation == 8:
            wm_matrice [:-pixel_translation, :, :] = wm_old_matrice[:-pixel_translation, :, :]
            gm_matrice [:-pixel_translation, :, :] = gm_old_matrice[:-pixel_translation, :, :]
            csf_matrice [:-pixel_translation, :, :] = csf_old_matrice[:-pixel_translation, :, :]
            raw_matrice [:-pixel_translation, :, :] = raw_old_matrice[:-pixel_translation, :, :]
        if which_translation == 9:
            wm_matrice [:-pixel_translation, :-pixel_translation, :] = wm_old_matrice[:-pixel_translation, :-pixel_translation, :]
            gm_matrice [:-pixel_translation, :-pixel_translation, :] = gm_old_matrice[:-pixel_translation, :-pixel_translation, :]
            csf_matrice [:-pixel_translation, :-pixel_translation, :] = csf_old_matrice[:-pixel_translation, :-pixel_translation, :]
            raw_matrice [:-pixel_translation, :-pixel_translation, :] = raw_old_matrice[:-pixel_translation, :-pixel_translation, :]
        if which_translation == 10:
            wm_matrice [:-pixel_translation, :, :-pixel_translation] = wm_old_matrice[:-pixel_translation, :, :-pixel_translation]
            gm_matrice [:-pixel_translation, :, :-pixel_translation] = gm_old_matrice[:-pixel_translation, :, :-pixel_translation]
            csf_matrice [:-pixel_translation, :, :-pixel_translation] = csf_matrice[:-pixel_translation, :, :-pixel_translation]
            raw_matrice [:-pixel_translation, :, :-pixel_translation] = raw_old_matrice[:-pixel_translation, :, :-pixel_translation]
        if which_translation == 11:
            wm_matrice [:-pixel_translation, :-pixel_translation, :-pixel_translation] = wm_old_matrice[:-pixel_translation, :-pixel_translation, :-pixel_translation]
            gm_matrice [:-pixel_translation, :-pixel_translation, :-pixel_translation] = gm_old_matrice[:-pixel_translation, :-pixel_translation, :-pixel_translation]
            csf_matrice [:-pixel_translation, :-pixel_translation, :-pixel_translation] = csf_old_matrice[:-pixel_translation, :-pixel_translation, :-pixel_translation]
            raw_matrice [:-pixel_translation, :-pixel_translation, :-pixel_translation] = raw_old_matrice[:-pixel_translation, :-pixel_translation, :-pixel_translation]
        if which_translation == 12:
            wm_matrice [:, :-pixel_translation, :] = wm_old_matrice[:, :-pixel_translation, :]
            gm_matrice [:, :-pixel_translation, :] = gm_old_matrice[:, :-pixel_translation, :]
            csf_matrice [:, :-pixel_translation, :] = csf_old_matrice[:, :-pixel_translation, :]
            raw_matrice [:, :-pixel_translation, :] = raw_old_matrice[:, :-pixel_translation, :]
        if which_translation == 13:
            wm_matrice [:, :-pixel_translation, :-pixel_translation] = wm_old_matrice[:, :-pixel_translation, :-pixel_translation]
            gm_matrice [:, :-pixel_translation, :-pixel_translation] = gm_old_matrice[:, :-pixel_translation, :-pixel_translation]
            csf_matrice [:, :-pixel_translation, :-pixel_translation] = csf_old_matrice[:, :-pixel_translation, :-pixel_translation]
            raw_matrice [:, :-pixel_translation, :-pixel_translation] = raw_old_matrice[:, :-pixel_translation, :-pixel_translation]
        if which_translation == 14:
            wm_matrice [:, :, :-pixel_translation] = wm_old_matrice[:, :, :-pixel_translation]
            gm_matrice [:, :, :-pixel_translation] = gm_old_matrice[:, :, :-pixel_translation]
            csf_matrice [:, :, :-pixel_translation] = csf_old_matrice[:, :, :-pixel_translation]
            raw_matrice [:, :, :-pixel_translation] = raw_old_matrice[:, :, :-pixel_translation]  
    else :
        wm_matrice = nib.load(path['wm']).get_data()
        gm_matrice = nib.load(path['gm']).get_data()
        csf_matrice = nib.load(path['csf']).get_data()
        raw_matrice = nib.load(path['raw_proc']).get_data()

    return (wm_matrice, gm_matrice, csf_matrice, raw_matrice)


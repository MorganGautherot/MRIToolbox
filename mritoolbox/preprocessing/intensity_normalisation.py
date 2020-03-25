def mri_normalisation(T1_mni, wm_mask, gm_mask, brain_mask, brain_inter_img_path):
	
	img_nifti = nib.load(T1_mni)
	img_affine = img_nifti.affine
	img_header = img_nifti.header
	img = img_nifti.get_data()

	wm = nib.load(wm_mask).get_data()
	gm = nib.load(gm_mask).get_data()
	brain_mask = nib.load(brain_mask).get_data()

	img_brain = img * brain_mask
	img_brain = img

	wm_mask = np.where(wm >= 0.8, 1, 0)
	gm_mask = np.where(gm >= 0.8, 1, 0)

	img_wm = img * wm_mask
	img_gm = img * gm_mask

	sns.kdeplot(img_brain.flatten()[img_brain.flatten()>0], shade=True, color='skyblue')
	plt.show()


	density_gm = sns.kdeplot(img_gm.flatten(), shade=True).get_lines()[0].get_data()

	gm_peak_value = np.sort(np.array(density_gm)[1, find_peaks(density_gm[1])[0]])[-1]
	gm_peak_index = np.squeeze(np.where(density_gm[1] == gm_peak_value))
	gm_peak = np.array(density_gm)[:, gm_peak_index]

	plt.scatter(gm_peak[0], gm_peak[1], color='red')
	plt.show()

	density_wm = sns.kdeplot(img_wm.flatten(), shade=True).get_lines()[0].get_data()

	wm_peak_value = np.sort(np.array(density_wm)[1, find_peaks(density_wm[1])[0]])[-1]
	wm_peak_index = np.squeeze(np.where(density_wm[1] == wm_peak_value))
	wm_peak = np.array(density_wm)[:, wm_peak_index]

	plt.scatter(wm_peak[0], wm_peak[1], color='red')

	plt.show()

	img_brain = np.where(img_brain<=0, 0, img_brain)
	brain_inter_img = 0.25 * (img_brain - gm_peak[0])/(wm_peak[0]-gm_peak[0]) + 0.75
	brain_inter_img = np.where(brain_inter_img>=1.5, 1.5, brain_inter_img)

	sns.kdeplot(brain_inter_img.flatten()[brain_inter_img.flatten()>0], shade=True)
	plt.show()

	img_nii = nib.Nifti1Image(brain_inter_img, affine=img_affine, header=img_header)

	nib.save(img_nii, brain_inter_img_path)
                

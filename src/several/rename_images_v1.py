import os 

class RenameFile:
    """Read two dirs: bw images and masks from these images to rename without names exports from PS
    """

    def __init__(self, images_dir, masks_dir):
        # get all file names (bw and masks)
        self.old_bw_names = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        self.old_mask_names = [f for f in os.listdir(masks_dir) if os.path.isfile(os.path.join(masks_dir, f))]

        # check if exist someone without right name
        idxs_imgs = self.checkintegrity(self.old_bw_names)
        idxs_mask = self.checkintegrity(self.old_mask_names)

        assert idxs_imgs.__len__() == 0
        assert idxs_mask.__len__() == 0

        # rename with the number and ONH prefix
        self.imgs_name = self.renamenames(self.old_bw_names)
        all_masks_name = self.renamenames(self.old_mask_names)

        # select only masks with the same same of bw imgs
        self.getmasks(all_masks_name)
        
        assert self.imgs_name.__len__() == self.selected_masks.__len__()

        # from all masks, select also only the masks with the same name of bw
        # but now with the original name
        self.selected_old_masks_name = self.select_orig_masks(self.old_mask_names, self.selected_masks)

        # rename files and put in another folder
        self.rename(masks_dir, self.selected_old_masks_name, self.selected_masks)
        self.rename(images_dir, self.old_bw_names, self.imgs_name)

    def checkintegrity(self, list) -> list:
        """Check if exists some image with the prefix "ONH" wrong, like "ON"

        Args
            list: a list with the name of images (or masks)
        """
        idxs = []

        for idx, i in enumerate(list):
            if "ONH" not in i:
                idxs.append(idx)

        return idxs

    def renamenames(self, list) -> list:
        """Rename all file names to the number and prefix "ONH" only
        
        Args
            list: a list with the name of images (or masks)
        """
        temp_list = []

        for i in list:
            pos = i.index("ONH")
            temp_list.append(i[0 : pos + 3] + '.png')

        return temp_list

    def getmasks(self, all_masks_names):
        """Select masks with the same id from list of bw images"""
        self.selected_masks = []

        for mask in all_masks_names:
            if mask in self.imgs_name:
                self.selected_masks.append(mask)

    def select_orig_masks(self, old_names, new_names):
        """Select only the original masks names that is compatible with `selected_masks`"""
        selected_old_names = []

        for mask in new_names:
            finded = False
            mask = mask[0:-4]
            for old_name in old_names:
                if old_name.find(mask) != -1:
                    selected_old_names.append(old_name)
                    finded = True
                    
                if finded: break

        return selected_old_names

    def rename(self, dir, list_old_names, list_new_names):

        newdir = dir + '\export'

        oldname = [os.path.join(dir, image_id) for image_id in list_old_names]
        newname = [os.path.join(newdir, image_id) for image_id in list_new_names]
        
        for idx in range(oldname.__len__()):
            os.rename(oldname[idx], newname[idx])

def main():

    RenameFile(
        images_dir=r"C:\Users\gfernandes\Desktop\imgs-claras",
        masks_dir=r"C:\Users\gfernandes\Desktop\masks"
    )

if __name__ == "__main__":
    main()

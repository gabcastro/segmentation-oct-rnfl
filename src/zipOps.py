from logging import error
import shutil
from zipfile import BadZipFile, ZipFile

class ZipOps:
    """Extract or compress a specific file"""

    def extractFile(self, fromDir, toDir):
        """Extract a file to a specific directory

        Args:
            fromDir: absolute path of file
            toDir: path where the file will be extract
        """ 

        try:
            zipContent = ZipFile(fromDir)
            zipContent.extractAll(toDir)
            zipContent.close()
        except BadZipFile as e:
            raise error(f'There are some problem with the zip, check the message: {e}')

    def compressFile(self, rootDir, baseDir, zipName):        
        """Compress a file to a specific directory

        Args:
            rootDir: directory that will be the root directory of the archive
            baseDir: will be the common prefix of all files and directories in the archive
            zipName: name of zip that will be created
        """
        shutil.make_archive(
            base_name=zipName,
            format='zip',
            root_dir=rootDir,
            base_dir=baseDir
        )
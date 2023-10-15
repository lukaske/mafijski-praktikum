cp ./MFPR/index.html ./dist/MFPR
cp ./MFPR/styles.css ./dist/MFPR
rsync -R ./MFPR/**/*.html ./dist
rsync -R ./MFPR/**/**/*.jpg ./dist
rsync -R ./MFPR/**/**/*.png ./dist
rsync -R ./MFPR/**/**/*.JPG ./dist
rsync -R ./MFPR/**/**/*.jpeg ./dist
rsync -R ./MFPR/**/**/*.JPEG ./dist
rsync -R ./MFPR/**/**/*.mov ./dist
rsync -R ./MFPR/**/**/*.png ./dist
rsync -R ./MFPR/**/**/*.PNG ./dist
rsync -R ./MFPR/**/**/*.gif ./dist
rsync -R ./MFPR/**/**/*.GIF ./dist
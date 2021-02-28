mkdir $2

mkdir $2/ckpts
ckpt_files=$(gsutil ls $1/ckpts)
for f in $ckpt_files; do
    gsutil cp $f $2/ckpts
done



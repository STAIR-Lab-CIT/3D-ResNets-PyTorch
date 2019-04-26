docker run --runtime=nvidia \
       --privileged \
       -v $(pwd):/3D-ResNets-PyTorch \
       -v /home1:/data4  \
       --workdir="/3D-ResNets-PyTorch" \
       --rm \
       -ti \
       --shm-size 32G \
       actrec/pip bash

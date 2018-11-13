Terraform Polyaxon on Google Kubernetes Engine
----------------------------------------------

This directory contains scripts to launch a robust Polyaxon instance using
Hashicorp Terraform. It's still a work in progress. The code should work on
OSX, Linux and Windows Subsystem for Linux.

1. Install the following on your development machine:

    * gcloud
    * kubectl: `gcloud components install kubectl`
    * helm
    * terraform

2. Launch a single-node file-server, named `polyaxon-nfs`:
   https://console.cloud.google.com/launcher/details/click-to-deploy-images/singlefs

This is the only bit of infrastructure that I haven't gotten into the Terraform
file yet.

3. Create your cluster, choosing a region and project name.

    terraform init    
    terraform apply


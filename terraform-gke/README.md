Terraform Polyaxon on Google Kubernetes Engine
----------------------------------------------

This directory contains scripts to launch a robust Polyaxon instance using
Hashicorp Terraform. It's still a work in progress. The code should work on
OSX, Linux and Windows Subsystem for Linux.

**1. Install the following on your development machine:**

* gcloud
* kubectl: `gcloud components install kubectl`
* helm
* terraform

**2. Launch a [single-node file-server](https://console.cloud.google.com/launcher/details/click-to-deploy-images/singlefs), named `polyaxon-nfs`.**

**3. Create your cluster, choosing a region and project name.**

```bash
terraform init    
terraform apply
```

If you're running this multiple times, you might want to edit your settings into the `main.tf` file. The `main.tf` file contains Terraform declarations for the cluster and its node pools. We define one node pool for the control nodes, and another for CPU workers. You can also define further node pools, e.g. for GPU workers. After creating the cluster, Terraform will fill some configuration values into the templates, execute a script to install Polyaxon.

**4. Install the bootstrap script (optional), and run it:**

```
sudo cp scripts/create_project.sh /usr/local/bin/create-polyaxon-project # Optional, but convenient.
mkdir -p ~/polyaxon-projects # Example -- could be anywhere
cd ~/polyaxon-projects
create-polyaxon-project my-new-project
cd my-new-project
source login.sh
```

The login.sh script will activate a virtualenv in your project directory, and
authenticate you with the Polyaxon cluster. You can run the
`create-polyaxon-project` command to create new projects as you need them.

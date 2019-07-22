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

**2. Generate an SSL certificate

**3. Create your cluster, choosing a region, hostname and project name.**

```bash
terraform init    
terraform apply
```

If you're running this multiple times, you might want to edit your settings into the `main.tf` file. The `main.tf` file contains Terraform declarations for the cluster and its node pools. We define one node pool for the control nodes, and another for CPU workers. You can also define further node pools, e.g. for GPU workers. After creating the cluster, Terraform will fill some configuration values into the templates, execute a script to install Polyaxon.

To delete the cluster, run:

```bash

terraform destroy
```

**4. Create a new project**

```
create-polyaxon-project ~/my-new-project
cd my-~/new-project
source login.sh
```

The login.sh script will activate a virtualenv in your project directory, and authenticate you with the Polyaxon cluster.

**5. Install the create-polyaxon-project script (optional)**

You might want to install the `create-polyaxon-project` somewhere on your `PATH`, to make it easy to run in future.

```
sudo cp scripts/create_project.sh /usr/local/bin/create-polyaxon-project # Optional, but convenient.
mkdir -p ~/polyaxon-projects # Example -- could be anywhere
cd ~/polyaxon-projects
```

You can run the `create-polyaxon-project` command to create new projects as you need them.

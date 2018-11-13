variable "project" {
    type = "string"
}

variable "zone" {
    type = "string"
}

provider "google" {
  project     = "${var.project}"
  zone = "${var.zone}"
}


resource "google_container_node_pool" "control-nodes" {
    name       = "polyaxon-control-nodes"
    cluster    = "${google_container_cluster.cluster.name}"
    node_count = 3
    node_config {
        preemptible  = true
        machine_type = "n1-standard-1"
        oauth_scopes = ["compute-rw", "storage-ro", "logging-write", "monitoring"]
    }
}


resource "google_container_cluster" "cluster" {
    name = "polyaxon-cluster"

    # Keep the default pool empty, and define node pools separately.
    lifecycle {
        ignore_changes = ["node_pool"]
    }
    node_pool {
        name = "default-pool"
    }
}


data "template_file" "install_polyaxon" {
    template = "${file("${path.module}/install-polyaxon.sh.tmpl")}"
    
    vars {
        project = "${var.project}"
        zone = "${var.zone}"
        name = "${google_container_cluster.cluster.name}"
    }
}

resource "null_resource" "trigger_script" {
    provisioner "local-exec" {
        command = "echo \"${data.template_file.install_polyaxon.rendered}\" > script.sh && bash ./script.sh"
    }
    depends_on = ["google_container_cluster.cluster"]
}

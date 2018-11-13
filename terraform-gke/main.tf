variable "project" {
    type = "string"
}

variable "zone" {
    type = "string"
}

variable "name" {
    type = "string"
    default = "polyaxon-cluster"
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
    name = "${var.name}"

    # Keep the default pool empty, and define node pools separately.
    lifecycle {
        ignore_changes = ["node_pool"]
    }
    node_pool {
        name = "default-pool"
    }
}


resource "null_resource" "trigger_script" {
    provisioner "local-exec" {
        command = "./scripts/install-polyaxon ${var.project} ${var.zone} ${var.name}"
    }

    depends_on = ["google_container_cluster.cluster"]
}

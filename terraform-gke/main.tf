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

resource "random_string" "polyaxon_password" {
    length = "16"
}

provider "google" {
  project     = "${var.project}"
  zone = "${var.zone}"
}

resource "google_container_node_pool" "core-nodes" {
    name       = "core"
    cluster    = "${google_container_cluster.cluster.name}"
    node_count = 3
    node_config {
        preemptible  = true
        machine_type = "n1-standard-4"
        oauth_scopes = ["compute-rw", "storage-ro", "logging-write", "monitoring"]
    }
}

resource "google_container_node_pool" "experiment-nodes" {
    name       = "experiments"
    cluster    = "${google_container_cluster.cluster.name}"
    node_count = 3
    node_config {
        preemptible  = true
        machine_type = "n1-standard-2"
        oauth_scopes = ["compute-rw", "storage-ro", "logging-write", "monitoring"]
    }
}

resource "google_container_node_pool" "build-nodes" {
    name       = "builds"
    cluster    = "${google_container_cluster.cluster.name}"
    node_count = 1
    node_config {
        preemptible  = true
        machine_type = "n1-standard-2"
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
        command = "./scripts/install-polyaxon ${var.project} ${var.zone} ${var.name} \"${random_string.polyaxon_password.result}\""
    }

    depends_on = [
        "google_container_cluster.cluster",
        "google_container_node_pool.core-nodes",
        "google_container_node_pool.build-nodes",
        "google_container_node_pool.experiment-nodes",
    ]
}

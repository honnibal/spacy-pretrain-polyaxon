#!/usr/bin/env bash

project_name=$1

# Create a project folder
mkdir -p $project_name
cd $project_name
python3 -m venv .env
.env/bin/pip install -r polyaxon-cli polyaxon-helper


# Write a script login.sh into the directory, that they can source. 
cat << EOF > login.sh

#!/usr/bin/env bash
source .env/bin/activate

export POLYAXON_IP=$(kubectl get svc --namespace polyaxon polyaxon-polyaxon-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export POLYAXON_HTTP_PORT=80
export POLYAXON_WS_PORT=80

echo http://$POLYAXON_IP:$POLYAXON_HTTP_PORT

polyaxon config set --host=$POLYAXON_IP --http_port=$POLYAXON_HTTP_PORT  --ws_port=$POLYAXON_WS_PORT
polyaxon config set --host=$POLYAXON_IP --http_port=$POLYAXON_HTTP_PORT  --ws_port=$POLYAXON_WS_PORT
polyaxon login --username=root --password="$(kubectl get secret --namespace polyaxon polyaxon-polyaxon-secret -o jsonpath="{.data.POLYAXON_ADMIN_PASSWORD}" | base64 --decode)"

EOF

source login.sh

polyaxon init $project_name
polyaxon create --project $project_name

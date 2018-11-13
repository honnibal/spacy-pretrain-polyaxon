#!/usr/bin/env bash
export POLYAXON_IP=$(kubectl get svc --namespace polyaxon polyaxon-polyaxon-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export POLYAXON_HTTP_PORT=80
export POLYAXON_WS_PORT=80

echo http://$POLYAXON_IP:$POLYAXON_HTTP_PORT

polyaxon config set --host=$POLYAXON_IP --http_port=$POLYAXON_HTTP_PORT  --ws_port=$POLYAXON_WS_PORT
polyaxon config set --host=$POLYAXON_IP --http_port=$POLYAXON_HTTP_PORT  --ws_port=$POLYAXON_WS_PORT
polyaxon login --username=root --password="$(kubectl get secret --namespace polyaxon polyaxon-polyaxon-secret -o jsonpath="{.data.POLYAXON_ADMIN_PASSWORD}" | base64 --decode)"


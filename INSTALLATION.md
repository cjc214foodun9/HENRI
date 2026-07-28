# TrustGraph Deployment Guide


## Platform Setup


### Install and configure Docker Compose

You need to have Docker Compose installed. See [Installing Docker Compose](https://docs.docker.com/compose/install/).


## Identity & Access Management


### Configure IAM bootstrap token

TrustGraph 2.4 introduces IAM (Identity and Access Management) for API and UX authentication. You must configure a bootstrap token to enable initial access. Set the `IAM_BOOTSTRAP_TOKEN` environment variable before starting the deployment. The token must have a `tg_` prefix to be recognised as an API token.

```
IAM_BOOTSTRAP_TOKEN=tg_your-secret-token-here
```


## Model Configuration


### Set up Ollama model server

The power of Ollama is the flexibility it provides in Language Model deployments. Being able to run LMs with Ollama enables fully secure AI TrustGraph pipelines that aren't relying on any external APIs. No data is leaving the host environment or network.

The Ollama service must be running, and have required models available using `ollama pull`. The Ollama service URL must be provided in an environment variable.

```
OLLAMA_HOST=http://ollama-host:11434
```

Replace the URL with the URL of your Ollama service.


## API Gateway


### Configure API gateway

The API Gateway is a required component which supports the CLI and Workbench. As of TrustGraph 2.4, gateway authentication is managed through IAM. No separate gateway secret is required. Ensure your `IAM_BOOTSTRAP_TOKEN` environment variable is set (see IAM setup).


### MCP server information

The MCP server allows MCP clients to interact with TrustGraph. As of TrustGraph 2.4, MCP server authentication is managed through IAM. No separate MCP server credentials are required.


## Deployment


### Deploy with Docker Compose

When you download the deploy configuration, you will have a ZIP file containing all the configuration needed to launch TrustGraph in Docker Compose. Unzip the ZIP file:

```bash
unzip deploy.zip
```

On MacOS, it may be necessary to specify a destination directory for the TrustGraph package:

```bash
unzip deploy.zip -d deploy
```

Navigate to the `docker-compose` directory. From this directory, launch TrustGraph with:

```bash
docker compose -f docker-compose.yaml up -d
```

If you are on Linux, running SELinux, you may need to change permissions on files in the deploy bundle so that they are accessible from within containers. This affects the `grafana` and `prometheus` directories.

```bash
chcon -Rt svirt_sandbox_file_t grafana prometheus
chmod 755 prometheus/ grafana/ grafana/*/
chmod 644 prometheus/* grafana/*/*
```


## Verification & Testing


### Access the TrustGraph Workbench

Once the system is running, you can access the Workbench on port 8888, or access using the following URL:

[http://localhost:8888/](http://localhost:8888/)

Once you have data loaded, you can present a Graph RAG query on the Chat tab. As well as answering the question, a list of semantic relationships which were used to answer the question are shown and these can be used to navigate the knowledge graph.


### Test Document RAG

Document RAG APIs are separate from GraphRAG. You can use `tg-invoke-document-rag` to test Document RAG processing once documents are loaded:

```bash
tg-invoke-document-rag -q "Describe a cat"
```

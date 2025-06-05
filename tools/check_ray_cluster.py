from loguru import logger


def get_ray_cluster_resources():
    """
    Get the total and per-node resources in a Ray cluster

    Returns:
        dict: A dictionary containing:
            - total_gpus: Total number of GPUs in the cluster
            - total_cpus: Total number of CPUs in the cluster
            - nodes: List of dictionaries containing per-node resources
                - node_id: Ray node ID
                - gpus: Number of GPUs on this node
                - cpus: Number of CPUs on this node
    """
    try:
        import ray
        ray.init()

        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()

        # Get node information
        nodes = []
        for node_info in ray.nodes():
            if node_info['Alive']:
                node_resources = {
                    'node_id': node_info['NodeID'],
                    'gpus': int(node_info['Resources'].get('GPU', 0)),
                    'cpus': int(node_info['Resources'].get('CPU', 0))
                }
                nodes.append(node_resources)
        return {
            'total_gpus': int(cluster_resources.get('GPU', 0)),
            'total_cpus': int(cluster_resources.get('CPU', 0)),
            'available_gpus': int(available_resources.get('GPU', 0)),
            'available_cpus': int(available_resources.get('CPU', 0)),
            'nodes': nodes
        }
    except Exception as e:
        logger.warning(f'Failed to get Ray cluster resources: {str(e)}')
        return None


def main():
    logger.info(get_ray_cluster_resources())


if __name__ == '__main__':
    main()

#include <cuda.h>
#include <stdint.h>

namespace pycuda
{
  struct CudaGraph{
    CUgraph_st graph;

    CudaGraph(unsigned int flags=0){ 
      CUDAPP_CALL_GUARDED(cuGraphCreate, (&graph, flags)); 
      }
    
    inline void graph_destroy(){
      CUDAPP_CALL_GUARDED(cuGraphDestroy,(graph)); 
    }

    inline void node_destroy(CUgraphNode node){
      CUDAPP_CALL_GUARDED(cuGraphDestroyNode,(node));
    }
    
    inline CudaGraph graph_clone(){
      CudaGraph graph_clone;
      CUDAPP_CALL_GUARDED(cuGraphClone,(&graph_clone, graph));
      return graph_clone;
    }
    
    inline CUgraphNode get_root_nodes(unsigned int numNodes){
      CUgraphNode nodes;
      CUDAPP_CALL_GUARDED(cuGraphGetNodes, (graph, &nodes, &numNodes));
      return nodes;
    }

    inline CUgraphNode get_root_nodes(unsigned int numRootNodes){
      CUgraphNode root_nodes;
      CUDAPP_CALL_GUARDED(cuGraphGetRootNodes, (graph, &root_nodes, &numRootNodes));
      return root_nodes;
    }

    inline void add_dependencies(const CUgraphNode from_nodes, const CUgraphNode to_nodes, unsigned int numdependencies=0){
      CUDAPP_CALL_GUARDED(cuGraphAddDependencies,(graph, &from_nodes, &to_nodes, numdependencies));
    }

    std::tuple <CudaGraphNode, CudaGraphNode> get_edges(unsigned int numdependencies=0){
      const CUgraphNode from_nodes;
      const CUgraphNode to_nodes;
      CUDAPP_CALL_GUARDED(cuGraphGetEdges,(graph, &from_nodes, &to_nodes, numdependencies));
      return make_tuple(from_nodes, to_nodes);
    }

    inline CUgraphNode get_dependencies(CUgraphNode node, unsigned int numdependencies=0){
      CUgraphNode dependencies;
      CUDAPP_CALL_GUARDED(cuGraphNodeGetDependencies, (node, &dependencies, &numdependencies));
      return dependencies;
    }

    inline CUgraphNode get_dependent_nodes(CUgraphNode node, unsigned int numdependentnodes=0){
      CUgraphNode dependent_nodes;
      CUDAPP_CALL_GUARDED(cuGraphNodeGetDependentNodes, (node, &dependent_nodes, &numdependentnodes));
      return dependent_nodes;
    }


    inline CUgraphNode add_child_graph_node(const CUgraphNode dependencies, unsigned int numdependencies=0, CudaGraph child_graph){
      CUgraphNode child_node;
      CUDAPP_CALL_GUARDED(cuGraphAddChildGraphNode,(&child_node, graph, &dependencies, numdependencies, child_graph));
      return child_node;
    }

    inline CUgraphNode add_empty_node(const CUgraphNode *dependencies, unsigned int numdependencies=0){
      CUgraphNode empty_node;
      CUDAPP_CALL_GUARDED(cuGraphAddEmptyNode,(*empty_node, graph, dependencies, numdependencies));
      return empty_node;
    }

    inline CUgraphNode add_event_record_node(Event event, const CUgraphNode dependencies, unsigned int numdependencies=0){
      CUgraphNode event_record_node;
      CUDAPP_CALL_GUARDED(cuGraphAddEventRecordNode,(&event_record_node, graph, &dependencies, numdependencies, event));
      return event_record_node;
    }

    inline CUgraphNode add_event_wait_node(Event event, const CUgraphNode dependencies, unsigned int numdependencies=0){
      CUgraphNode event_wait_node;
      CUDAPP_CALL_GUARDED(cuGraphAddEventWaitNode,(&event_record_node, graph, &dependencies, numdependencies, event));
      return event_wait_node;
    }

    inline Event get_event_from_event_record_node(CUgraphNode event_record_node){
      CUevent event;
      CUDAPP_CALL_GUARDED(cuGraphEventRecordNodeGetEvent,(&event_record_node, &event));
      return event
    }

    inline Event get_event_from_event_wait_node(CUgraphNode event_wait_node){
      CUevent event;
      CUDAPP_CALL_GUARDED(cuGraphEventWaitNodeGetEvent,(event_wait_node, &event));
      return event
    }

    inline void set_record_node_event(ExecGraph graph_exec, CUgraphNode event_record_node, CUevent event){
      CUDAPP_CALL_GUARDED(cuGraphEventWaitNodeSetEvent,(graph_exec, event_record_node, event));
    }

    inline void set_wait_node_event(ExecGraph graph_exec, CUgraphNode event_wait_node, CUevent event){
      CUDAPP_CALL_GUARDED(cuGraphEventWaitNodeSetEvent,(graph_exec, event_wait_node, event));
    }

    inline CUgraphNode add_host_node(const CUgraphNode dependencies, unsigned int numdependencies=0, const CUDA_HOST_NODE_PARAMS &nodeParams){
      CUgraphNode host_node;
      CUDAPP_CALL_GUARDED(cuGraphAddHostNode,(&host_node, graph, &dependencies, numdependencies, &nodeParams));
      return host_node;
    }


  }

}


  
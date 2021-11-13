from __future__ import annotations
from typing import Iterable, Sequence


class CudaGraphNode:
    def get_dependenices(self) -> List[CudaGraphNode]:
        ...
        
    def get_dependendent_nodes(self) -> List[CudaGraphNode]:
        ...
        

class HostNodeParams:
    ...


class CudaGraph:
    def add_child_graph(self,
                        dependencies: Iterable[CudaGraphNode],
                        child_graph: CudaGraph) -> CudaGraphNode:
        ...

    def add_child_graph(self,
                        dependencies: Iterable[CudaGraphNode],
                        child_graph: CudaGraph) -> CudaGraphNode:
        ...
    
    def create_graph(self, flags: int): -> CudaGraph
        ...

    def clone_graph(self) -> CudaGraph:
        ...
    

    def create_exec_graph(self,
                          graph: CudaGraph): -> CudaExecutableGraph
        ...
    
    def debug_dot_print(self,
                        graph: CudaGraph,
                        path : const char*
                        flags : unsigned int): -> None
        ...
        
    def add_dependencies(self,
                         from_: Sequence[CudaGraphNode],
                         to: Sequence[CudaGraphNode]) -> None:
        if len(from_) != len(to):
            raise ValueError(...)
        ...

    def add_empty_node(self,
                       dependencies: Iterable[CudaGraphNode]) -> CudaGraphNode:
        ...

    def add_event_record_node(self,
                              dependencies: Iterable[CudaGraphNode],
                              event: Event) -> CudaGraphNode:
        ...
    
    def get_event_from_event_record_node(self,
                                         node: CudaGraphNode
                                         ) -> Event:
        ...
    
    def set_event_record_node_event(self,
                                    node: CudaGraphNode,
                                    event: Event) -> None
        ...
    
    def add_wait_node(self,
                      dependencies: Iterable[CudaGraphNode]) -> CudaGraphNode:
        ...
    
    def get_event_from_event_wait_node(self,
                                       node: CudaGraphNode
                                       ) -> Event
        ...
    
    def set_event_wait_node_event(self,
                            node: CudaGraphNode,
                            event: Event) -> None
        ...


    def add_kernel_node(self,
                       dependencies   : Iterable[CudaGraphNode],
                       blockDim       : dim3,
                       gridDim        : dim3,
                       sharedMemBytes : unsigned int,
                       func           : void) -> CudaGraphNode:
        ...

    def add_mem_alloc_node(self,
                       dependencies   : Iterable[CudaGraphNode],
                       accessDesCount : size_t,
                       accessDescs    : cudaMemAccessDesc ,
                       bytesize       : bytesize,
                       dptr           : void,
                       poolProps      : struct) -> CudaGraphNode:
        ...
    
    def add_memset_node(self,
                       dependencies: Iterable[CudaGraphNode],
                       dst         : void,
                       elementSize : unsigned int,
                       height      : size_t,
                       pitch       : size_t,
                       value       : unsigned int,
                       width       : size_t) -> CudaGraphNode:
        ...

class ExecGraph:
    def launch(self, stream: Stream) -> None
        ...
    
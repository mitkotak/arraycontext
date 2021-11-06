from __future__ import annotations
from typing import Iterable, Sequence


class CudaGraphNode:
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
    
    def create_graph(self): -> CudaGraph
        ...

    def clone_graph(self,
                    graph: CudaGraph) -> CudaGraph:
        ...
    

    def create_executable_graph(self,
                                graph: CudaGraph): -> CudaExecutableGraph
        ...
    
    def launch_executable_graph(self,
                                graph_executable: CudaExecutableGraph,
                                stream: CudaStream) -> None
        ...

    def remove_executable_graph(self,
                                graph_executable: CudaExecutableGraph) -> None
        ...

    def add_dependencies(self,
                         from_: Sequence[CudaGraphNode],
                         to: Sequence[CudaGraphNode]) -> None:
        if len(from_) != len(to):
            raise ValueError(...)
        ...
    
    def get_dependencies(self,
                         from_: Sequence[CudaGraphNode],
                         to: Sequence[CudaGraphNode]) -> Iterable[CudaGraphNode]:
        ...
        
    def remove_node(self,
                    node: CudaGraphNode) -> None
        ...
    
    def add_empty_node(self,
                       dependencies: Iterable[CudaGraphNode]) -> CudaGraphNode:
        ...

    def add_event_record_node(self,
                              dependencies: Iterable[CudaGraphNode]) -> CudaGraphNode:
        ...
    
    def get_event_event_record_node(self,
                                    node: CudaGraphNode) -> *CudaEvent
        ...
    
    def set_event_event_record_node(self,
                                    node: CudaGraphNode, event: *CudaEvent) -> None
        ...
    
    def add_wait_node(self,
                      dependencies: Iterable[CudaGraphNode]) -> CudaGraphNode:
        ...
    
    def get_event_wait_node(self,
                            node: CudaGraphNode) -> *CudaEvent
        ...
    
    def set_event_wait_node(self,
                            node: CudaGraphNode, event: *CudaEvent) -> None
        ...
    
    def add_external_semaphores_signal_node(self,
                                            dependencies: Iterable[CudaGraphNode]) -> CudaGraphNode:
        ...
    
    def add_external_semaphores_wait_node(self,
                                            dependencies: Iterable[CudaGraphNode]) -> CudaGraphNode:
        ...
    
    def add_host_node(self,
                       dependencies: Iterable[CudaGraphNode]) -> CudaGraphNode:
        ...

    def add_kernel_node(self,
                       dependencies: Iterable[CudaGraphNode]) -> CudaGraphNode:
        ...

    def add_mem_alloc_node(self,
                       dependencies: Iterable[CudaGraphNode]) -> CudaGraphNode:
        ...
    
    def add_memset_node(self,
                       dependencies: Iterable[CudaGraphNode]) -> CudaGraphNode:
        ...
    
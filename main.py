import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

import pandas as pd


class Graph:
    def __init__(self, vertices=None, directed=False):
        self.vertices = vertices if vertices else []
        self.edges = []
        self.directed = directed
        self.vertex_index = {v: i for i, v in enumerate(self.vertices)}
        self.adjacency = defaultdict(list)
        self.predecessor = defaultdict(list)

    def to_networkx(self):

        G = nx.DiGraph() if self.directed else nx.Graph()
        G.add_nodes_from(self.vertices)
        for edge in self.edges:
            if len(edge) == 3:
                G.add_edge(edge[0], edge[1], weight=edge[2])
            else:
                G.add_edge(edge[0], edge[1])
        return G
    def draw_dynamic_list_of_vertices(self):

        if not self.vertices:
            print("No vertices to draw dynamic list.")
            return

        plt.figure(figsize=(18, 18))  # Increased size for better detail
        ax = plt.gca()
        ax.axis('off')
        ax.set_title("1. Dynamic List of Vertices", fontsize=16, fontweight='bold')

        main_node_width = 0.8
        main_node_height = 0.5
        main_node_data_height = 0.3
        main_node_ptr_height = 0.2
        spacing_x = 1.5
        start_x = 0
        start_y_main_list = 3.5


        plt.text(start_x - 1, start_y_main_list + main_node_height / 2, 'G',
                 ha='center', va='center', fontsize=12, fontweight='bold', color='darkblue')
        plt.arrow(start_x - 0.7, start_y_main_list + main_node_height / 2,
                  0.5, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

        current_x = start_x
        main_list_nodes_pos = {}

        sorted_vertices = sorted(self.vertices)

        for i, vertex in enumerate(sorted_vertices):

            outer_rect = plt.Rectangle((current_x, start_y_main_list), main_node_width, main_node_height,
                                       facecolor='white', edgecolor='black', zorder=2)
            ax.add_patch(outer_rect)


            vertex_rect = plt.Rectangle((current_x, start_y_main_list + main_node_ptr_height), main_node_width,
                                        main_node_data_height,
                                        facecolor='white', edgecolor='black', zorder=3)
            ax.add_patch(vertex_rect)
            plt.text(current_x + main_node_width / 2,
                     start_y_main_list + main_node_ptr_height + main_node_data_height / 2,
                     str(vertex), ha='center', va='center', fontsize=12, fontweight='bold')


            plt.text(current_x + main_node_width * 0.75, start_y_main_list + main_node_ptr_height / 2,
                     '.', ha='center', va='center', fontsize=20, color='gray')


            main_list_nodes_pos[vertex] = (current_x + main_node_width / 2, start_y_main_list)


            if i < len(sorted_vertices) - 1:
                # Arrow from right edge of current node to left edge of next node
                plt.arrow(current_x + main_node_width, start_y_main_list + main_node_height / 2,
                          spacing_x - main_node_width, 0,
                          head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=1)
            else:
                # Explicit 'nul' for the last node's next pointer
                plt.text(current_x + main_node_width + 0.2, start_y_main_list + main_node_height / 2,
                         '.', ha='center', va='center', fontsize=20, color='gray')  # The 'dot' in the image

            current_x += spacing_x

        # --- Draw Successor Linked Lists ---
        succ_node_width = 0.8
        succ_node_height = 0.6  # Slightly taller for data and next pointer
        succ_node_data_height = 0.4
        succ_node_ptr_height = 0.2

        successor_spacing_y = 0.7  # Vertical spacing between successor nodes

        max_successor_y_depth = 0

        for vertex in sorted_vertices:
            sx, sy = main_list_nodes_pos[vertex]  # Get position of main vertex node (bottom center)

            successors = sorted(self.adjacency.get(vertex, []))

            # Draw pointer from main vertex node to its successor list head
            # The arrow originates from the lower left part of the main node to the first successor's top.
            arrow_start_x = sx - main_node_width / 2 + 0.1  # From near left corner of pointer section
            arrow_start_y = sy + main_node_ptr_height / 2

            if successors:
                first_succ_y = start_y_main_list - 1.0  # Initial y for the first successor node
                first_succ_x = sx - succ_node_width / 2  # Center it under the main node

                # Arrow from main node (bottom-left area) to first successor (top-center)
                # It goes down, then right to the first successor's box
                plt.annotate('', xy=(first_succ_x + succ_node_width / 2, first_succ_y + succ_node_height),
                             # Target (top center of first successor)
                             xytext=(arrow_start_x, arrow_start_y),  # Start (bottom-left of main node pointer section)
                             arrowprops=dict(arrowstyle='-|>', color='red', lw=1.5, mutation_scale=15,
                                             connectionstyle="arc3,rad=-0.3"))  # Curved arrow

                current_sy = first_succ_y  # Start y for drawing successor nodes

                for j, succ_vertex in enumerate(successors):

                    outer_rect_succ = plt.Rectangle((first_succ_x, current_sy), succ_node_width, succ_node_height,
                                                    facecolor='white', edgecolor='black', zorder=2)
                    ax.add_patch(outer_rect_succ)


                    succ_data_rect = plt.Rectangle((first_succ_x, current_sy + succ_node_ptr_height), succ_node_width,
                                                   succ_node_data_height,
                                                   facecolor='white', edgecolor='black', zorder=3)
                    ax.add_patch(succ_data_rect)
                    plt.text(first_succ_x + succ_node_width / 2,
                             current_sy + succ_node_ptr_height + succ_node_data_height / 2,
                             str(succ_vertex), ha='center', va='center', fontsize=10, fontweight='bold')


                    plt.text(first_succ_x + succ_node_width * 0.75, current_sy + succ_node_ptr_height / 2, '.',
                             ha='center', va='center', fontsize=20, color='gray')


                    if j < len(successors) - 1:
                        plt.arrow(first_succ_x + succ_node_width / 2, current_sy,
                                  0, -successor_spacing_y + succ_node_height,
                                  head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=1)
                    else:

                        plt.text(first_succ_x + succ_node_width / 2, current_sy - 0.3, 'N',
                                 ha='center', va='center', fontsize=10, fontweight='bold', color='gray')

                    current_sy -= successor_spacing_y

                max_successor_y_depth = max(max_successor_y_depth, start_y_main_list - current_sy)
            else:

                plt.text(sx - succ_node_width / 2 + succ_node_width / 2, start_y_main_list - 1.0, 'N', ha='center',
                         va='center', fontsize=10, fontweight='bold', color='gray')
                max_successor_y_depth = max(max_successor_y_depth, 1.0)  # Fixed depth for single 'N'


        ax.set_xlim(start_x - 1.5, current_x + 0.5)
        ax.set_ylim(start_y_main_list - max_successor_y_depth - 1.0,
                    start_y_main_list + 1.0)
        plt.show()


    def draw_dynamic_list_of_arcs(self):


        if not self.edges:
            print("No arcs (edges) to draw dynamic list.")
            return

        plt.figure(figsize=(18, 8))
        ax = plt.gca()
        ax.axis('off')
        ax.set_title("2. Dynamic List of Arcs", fontsize=16, fontweight='bold')

        arc_node_width = 0.8
        arc_node_height = 1.0
        arc_node_uv_height = 0.35
        arc_node_ptr_height = 0.3

        spacing_x = 1.5
        start_x = 0
        start_y = 2.0


        plt.text(start_x - 1, start_y + arc_node_height / 2, 'G',
                 ha='center', va='center', fontsize=12, fontweight='bold', color='darkblue')
        plt.arrow(start_x - 0.7, start_y + arc_node_height / 2,
                  0.5, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

        current_x = start_x


        sorted_edges = sorted(self.edges, key=lambda x: (x[0], x[1]))

        for i, edge in enumerate(sorted_edges):
            u, v = edge[0], edge[1]


            outer_rect = plt.Rectangle((current_x, start_y), arc_node_width, arc_node_height,
                                       facecolor='white', edgecolor='black', zorder=2)
            ax.add_patch(outer_rect)


            u_rect = plt.Rectangle((current_x, start_y + arc_node_ptr_height + arc_node_uv_height), arc_node_width,
                                   arc_node_uv_height,
                                   facecolor='white', edgecolor='black', zorder=3)
            ax.add_patch(u_rect)
            plt.text(current_x + arc_node_width / 2,
                     start_y + arc_node_ptr_height + arc_node_uv_height + arc_node_uv_height / 2,
                     str(u), ha='center', va='center', fontsize=12, fontweight='bold')


            v_rect = plt.Rectangle((current_x, start_y + arc_node_ptr_height), arc_node_width, arc_node_uv_height,
                                   facecolor='white', edgecolor='black', zorder=3)
            ax.add_patch(v_rect)
            plt.text(current_x + arc_node_width / 2, start_y + arc_node_ptr_height + arc_node_uv_height / 2,
                     str(v), ha='center', va='center', fontsize=12, fontweight='bold')


            plt.text(current_x + arc_node_width * 0.75, start_y + arc_node_ptr_height / 2, '.',
                     ha='center', va='center', fontsize=20, color='gray')


            if i < len(sorted_edges) - 1:
                plt.arrow(current_x + arc_node_width, start_y + arc_node_height / 2,
                          spacing_x - arc_node_width, 0,
                          head_width=0.1, head_length=0.1, fc='black', ec='black', zorder=1)
            else:

                plt.text(current_x + arc_node_width + 0.2, start_y + arc_node_height / 2,
                         '.', ha='center', va='center', fontsize=20, color='gray')


            plt.text(current_x + arc_node_width / 2, start_y - 0.3,
                     f"U{i + 1}", ha='center', va='center', fontsize=10, color='darkgray')

            current_x += spacing_x


        ax.set_xlim(start_x - 1.5, current_x + 0.5)
        ax.set_ylim(start_y - 0.5, start_y + arc_node_height + 0.5)
        plt.show()
    def generate_successor_linear_list(self):
        """إنشاء القائمة الخطية للخلف بناءً على بيانات الرسم البياني
           مطابق لطريقة العرض في الصورة 7. Linear List of successors
        """
        elements = []
        pointers = {}
        current_pos = 1


        sorted_vertices_for_list = sorted(self.vertices)

        for vertex in sorted_vertices_for_list:
            successors = sorted(self.adjacency.get(vertex, []))
            if successors:
                pointers[vertex] = current_pos
                elements.extend(successors)
                current_pos += len(successors)
            else:
                pointers[vertex] = 'nul'

        return elements, pointers

    def generate_eol_linear_list(self):

        elements = []
        pointers = {}
        current_pos = 1

        sorted_vertices_for_list = sorted(self.vertices)

        for vertex in sorted_vertices_for_list:
            successors = sorted(self.adjacency.get(vertex, []))
            if successors:
                pointers[vertex] = current_pos
                elements.extend(successors)
                elements.append('$')
                current_pos += len(successors) + 1
            else:
                pointers[vertex] = current_pos
                elements.append('$')
                current_pos += 1

        return elements, pointers

    def generate_bol_linear_list(self):

        elements = []
        pointers = {}
        current_pos = 1

        sorted_vertices_for_list = sorted(self.vertices)

        for vertex in sorted_vertices_for_list:
            successors = sorted(self.adjacency.get(vertex, []))
            if successors:
                pointers[vertex] = current_pos
                elements.append('#')
                elements.extend(successors)
                current_pos += len(successors) + 1
            else:
                pointers[vertex] = current_pos
                elements.append('#')
                current_pos += 1

        return elements, pointers

    def _draw_linear_list_common(self, elements, pointers, title, pointer_color, eol_bol_char=None):

        if not elements and not pointers:
            print(f"No elements or pointers in the linear list for '{title}' to draw.")
            return


        max_elements_len = len(elements)
        if 'nul' in pointers.values():

            pass

        plt.figure(figsize=(max(12, max_elements_len * 1.0), 6))


        for i, val in enumerate(elements):
            x_pos = i + 1
            # Elements box
            plt.text(x_pos, 2, str(val), ha='center', va='center',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))
            # Index below element
            plt.text(x_pos, 1.5, str(x_pos), ha='left', va='baseline', fontsize=14, color='black')


        sorted_vertices_for_list = sorted(self.vertices)


        vertex_label_positions = {}
        current_bottom_x_offset = 1
        nul_pointers_count = 0
        actual_pointers_x_coords = sorted(list(set(p for p in pointers.values() if p != 'nul')))


        vertex_ptr_list = sorted([(v, p) for v, p in pointers.items()],
                                 key=lambda x: (x[1] if x[1] != 'nul' else float('inf'), x[0]))

        current_bottom_x_pos = 1
        x_step = 2.0


        if vertex_ptr_list and vertex_ptr_list[0][1] != 'nul':
            current_bottom_x_pos = max(1, vertex_ptr_list[0][1])


        bottom_label_x_map = {}
        current_x = 1
        for vertex in sorted_vertices_for_list:
            ptr = pointers.get(vertex, 'nul')
            if ptr != 'nul':

                bottom_label_x_map[vertex] = ptr
            else:

                last_element_x = max(elements_idx for elements_idx in range(1, len(elements) + 1)) if elements else 1

                if 'last_nul_x' in locals():
                    bottom_label_x_map[vertex] = last_nul_x + x_step
                else:

                    bottom_label_x_map[vertex] = last_element_x + x_step
                last_nul_x = bottom_label_x_map[vertex]


        for vertex in sorted_vertices_for_list:
            ptr = pointers.get(vertex, 'nul')


            bottom_x = bottom_label_x_map.get(vertex, 0)

            if ptr != 'nul':

                plt.annotate('', xy=(ptr, 1.8), xytext=(bottom_x, 0.8),  # Arrow from bottom label up to element
                             arrowprops=dict(arrowstyle='-|>', color=pointer_color, lw=1.5, mutation_scale=15))
                plt.text(bottom_x, 0.5, str(vertex), ha='center', va='center', fontsize=10,
                         color='black')  # Vertex label
                plt.text(bottom_x, 0.2, f"Ptr. {str(vertex)}", ha='center', va='center', fontsize=8,
                         color='black')  # "Ptr. a" label
            else:

                plt.text(bottom_x, 0.5, 'nul', ha='center', va='center', fontsize=10, color='gray',
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2'))
                plt.text(bottom_x, 0.2, f"Ptr. {str(vertex)}", ha='center', va='center', fontsize=8,
                         color='black')  # "Ptr. f" label


        max_x_for_elements = len(elements) + 1
        max_x_for_labels = max(bottom_label_x_map.values(), default=0) if bottom_label_x_map else 0
        plt.xlim(0.5, max(max_x_for_elements, max_x_for_labels) + 1)
        plt.ylim(0, 3)

        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

    def draw_linear_list_of_successors(self):

        elements, pointers = self.generate_successor_linear_list()


        self._draw_linear_list_common(elements, pointers, "Linear List of Successors", 'blue')

    def draw_linear_list_with_eol(self):

        elements, pointers = self.generate_eol_linear_list()
        self._draw_linear_list_common(elements, pointers, "Linear List of Successors with End of List (EOL)", 'green',
                                      eol_bol_char='$')

    def draw_linear_list_with_bol(self):

        elements, pointers = self.generate_bol_linear_list()
        self._draw_linear_list_common(elements, pointers, "Linear list of Successors with Beginning of List (BOL)",
                                      'purple', eol_bol_char='#')

    def add_vertex(self, vertex):
        if vertex not in self.vertex_index:
            self.vertex_index[vertex] = len(self.vertices)
            self.vertices.append(vertex)

    def add_edge(self, u, v, weight=None):
        self.add_vertex(u)
        self.add_vertex(v)
        edge = (u, v, weight) if weight else (u, v)
        self.edges.append(edge)
        self.adjacency[u].append(v)
        self.predecessor[v].append(u)
        if not self.directed:
            self.adjacency[v].append(u)
            self.predecessor[u].append(v)

    def adjacency_matrix(self):
        n = len(self.vertices)
        self.vertex_index = {v: i for i, v in enumerate(self.vertices)}
        matrix = np.zeros((n, n), dtype=int)

        for edge in self.edges:
            if len(edge) < 2:
                continue
            u, v = edge[0], edge[1]
            i, j = self.vertex_index[u], self.vertex_index[v]
            matrix[i][j] += 1
            if not self.directed:
                matrix[j][i] += 1

        return matrix

    def incidence_matrix(self):
        n = len(self.vertices)
        m = len(self.edges)
        matrix = np.zeros((n, m), dtype=int)
        for edge_idx, (u, v, *_) in enumerate(self.edges):
            i = self.vertex_index[u]
            j = self.vertex_index[v]
            matrix[i][edge_idx] = 1 if self.directed else 1
            matrix[j][edge_idx] = -1
        return matrix

    def successors_list(self):
        return dict(self.adjacency)

    def predecessors_list(self):
        return dict(self.predecessor)

    def vertices_linked_list(self):
        return {v: self.vertices[i + 1] if i + 1 < len(self.vertices) else None
                for i, v in enumerate(self.vertices)}

    def edges_linked_list(self):
        return {e: self.edges[i + 1] if i + 1 < len(self.edges) else None
                for i, e in enumerate(self.edges)}

    def vertex_coloring(self):
        colors = {}
        if not self.vertices:
            return colors

        colors[self.vertices[0]] = 0
        available = [False] * len(self.vertices)

        for v in self.vertices[1:]:
            for neighbor in self.adjacency[v]:
                if neighbor in colors:
                    available[colors[neighbor]] = True

            color = 0
            while color < len(self.vertices):
                if not available[color]:
                    break
                color += 1

            colors[v] = color
            available = [False] * len(self.vertices)

        return colors

    def edge_coloring(self):
        edge_colors = {}
        if not self.edges:
            return edge_colors

        max_degree = max(len(neighbors) for neighbors in self.adjacency.values()) if self.adjacency else 0
        colors = list(range(max_degree + 1))

        for u, v, *_ in self.edges:
            used_colors = set()

            for neighbor in self.adjacency[u]:
                if (u, neighbor) in edge_colors:
                    used_colors.add(edge_colors[(u, neighbor)])
                elif not self.directed and (neighbor, u) in edge_colors:
                    used_colors.add(edge_colors[(neighbor, u)])

            for neighbor in self.adjacency[v]:
                if (v, neighbor) in edge_colors:
                    used_colors.add(edge_colors[(v, neighbor)])
                elif not self.directed and (neighbor, v) in edge_colors:
                    used_colors.add(edge_colors[(neighbor, v)])

            for color in colors:
                if color not in used_colors:
                    edge_colors[(u, v)] = color
                    if not self.directed:
                        edge_colors[(v, u)] = color
                    break

        return edge_colors

    def draw_graph(self):
        try:
            G = nx.DiGraph() if self.directed else nx.Graph()

            if not self.vertices:
                print("Graph is empty - no vertices to draw")
                return

            G.add_nodes_from(self.vertices)

            for edge in self.edges:
                if len(edge) == 3:
                    G.add_edge(edge[0], edge[1], weight=edge[2])
                else:  # بدون وزن
                    G.add_edge(edge[0], edge[1])

            pos = nx.spring_layout(G, seed=42)

            plt.figure(figsize=(10, 8))
            nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue')
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

            if self.directed:
                nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20,
                                       edge_color='gray', width=1.5)
            else:
                nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5)

            plt.title("Graph Visualization")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error drawing graph: {e}")
            print("Vertices:", self.vertices)
            print("Edges:", self.edges)

    def draw_adjacency_matrix(self):
        matrix = self.adjacency_matrix()
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap='Blues', interpolation='none')

        for i in range(len(self.vertices)):
            for j in range(len(self.vertices)):
                plt.text(j, i, str(matrix[i][j]),
                         ha='center', va='center', color='black')

        plt.xticks(range(len(self.vertices)), self.vertices)
        plt.yticks(range(len(self.vertices)), self.vertices)
        plt.colorbar()
        plt.title("Adjacency Matrix")
        plt.show()

    def draw_incidence_matrix(self):
        matrix = self.incidence_matrix()
        plt.figure(figsize=(12, 6))
        plt.imshow(matrix, cmap='Reds', interpolation='none')

        for i in range(len(self.vertices)):
            for j in range(len(self.edges)):
                plt.text(j, i, str(matrix[i][j]),
                         ha='center', va='center', color='black')

        plt.xticks(range(len(self.edges)), [f'E{i}' for i in range(len(self.edges))])
        plt.yticks(range(len(self.vertices)), self.vertices)
        plt.colorbar()
        plt.title("Incidence Matrix")
        plt.show()




    def get_successors_table(self):
        return {v: sorted(succ) for v, succ in self.adjacency.items()}

    def get_predecessors_table(self):
        return {v: sorted(pred) for v, pred in self.predecessor.items()}  # التصحيح هنا

    def draw_table(self, title, data):
        if not self.vertices:
            print(f"No vertices to draw {title} table.")
            return

        vertices = sorted(self.vertices)

        max_len = 0
        for v in vertices:
            max_len = max(max_len, len(data.get(v, [])))


        table_data = []
        for i in range(max_len):
            row = []
            for v in vertices:
                items = data.get(v, [])
                row.append(items[i] if i < len(items) else "")
            table_data.append(row)

        df = pd.DataFrame(table_data, columns=[str(v) for v in vertices])

        fig, ax = plt.subplots(figsize=(max(8, len(vertices) * 1.5), max(4, max_len * 0.8)))  # Dynamic sizing
        ax.axis('off')

        if df.empty:
            ax.text(0.5, 0.5, "Table is empty.", ha='center', va='center', fontsize=12)
        else:
            table = ax.table(cellText=df.values,
                             colLabels=df.columns,
                             cellLoc='center',
                             loc='center',
                             bbox=[0, 0, 1, 1])

            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.2)


            for (row, col), cell in table._cells.items():
                if row == 0:
                    cell.set_facecolor("#40466e")
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    cell.set_facecolor("#f2f2f2")

        plt.title(title, fontweight='bold', pad=20, fontsize=16)
        plt.tight_layout()
        plt.show()

    def show_successors_table(self):
        data = self.get_successors_table()
        self.draw_table("c) Successors List", data)

    def show_predecessors_table(self):
        data = self.get_predecessors_table()
        self.draw_table("d) predecessors List", data)

    def greedy_vertex_coloring(self):

        if not self.vertices:
            return {}

        degrees = {v: len(self.adjacency[v]) for v in self.vertices}
        sorted_vertices = sorted(self.vertices, key=lambda x: -degrees[x])

        colors = {}
        color_groups = {}

        for vertex in sorted_vertices:
            used_colors = {colors[neighbor] for neighbor in self.adjacency[vertex] if neighbor in colors}


            for existing_color, group in color_groups.items():
                if all(neighbor not in self.adjacency[vertex] for neighbor in group):
                    colors[vertex] = existing_color
                    color_groups[existing_color].append(vertex)
                    break
            else:

                new_color = len(color_groups)
                colors[vertex] = new_color
                color_groups[new_color] = [vertex]

        self._validate_coloring(colors)
        return colors

    def _validate_coloring(self, colors):

        for u in self.vertices:
            for v in self.adjacency[u]:
                if colors[u] == colors[v]:
                    new_color = max(colors.values()) + 1
                    colors[v] = new_color
                    print(f"تم تصحيح التلوين: الرأس {v} حصل على لون جديد {new_color}")

        print("تم التحقق من التلوين بنجاح")

    def greedy_edge_coloring(self):

        if not self.edges:
            return {}

        edge_colors = {}
        color_usage = defaultdict(set)


        degrees = {v: len(self.adjacency[v]) for v in self.vertices}
        sorted_edges = sorted(self.edges, key=lambda e: -(degrees[e[0]] + degrees[e[1]]))

        for edge in sorted_edges:
            u, v = edge[0], edge[1]


            used_colors = color_usage[u].union(color_usage[v])


            color = 0
            while color in used_colors:
                color += 1

            edge_colors[(u, v)] = color
            color_usage[u].add(color)
            color_usage[v].add(color)

            if not self.directed:
                edge_colors[(v, u)] = color

        return edge_colors

    def draw_edge_coloring(self, edge_colors, title=""):

        G = self.to_networkx()
        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(10, 8))


        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightgray')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')


        edge_list = []
        color_list = []
        for edge in G.edges():
            edge_list.append(edge)
            if edge in edge_colors:
                color_list.append(edge_colors[edge])
            elif (edge[1], edge[0]) in edge_colors and not self.directed:
                color_list.append(edge_colors[(edge[1], edge[0])])
            else:
                color_list.append(0)

        nx.draw_networkx_edges(
            G, pos, edgelist=edge_list, edge_color=color_list,
            width=2, edge_cmap=plt.cm.tab10, edge_vmin=0, edge_vmax=9,
            arrows=self.directed, arrowsize=20
        )

        plt.title(f"Edge Coloring {title}", fontsize=16)
        plt.axis('off')
        plt.show()

    def draw_vertex_coloring(self, vertex_colors=None, title="Vertex Coloring"):

        if not self.vertices:
            print("الرسم البياني فارغ - لا يوجد رؤوس لعرضها")
            return

        G = self.to_networkx()
        pos = nx.spring_layout(G, seed=42)


        if vertex_colors is None:
            vertex_colors = self.greedy_vertex_coloring()


        color_map = []
        for node in G.nodes():
            color_map.append(vertex_colors.get(node, 0))

        plt.figure(figsize=(10, 8))


        nx.draw_networkx_nodes(
            G, pos,
            node_size=800,
            node_color=color_map,
            cmap=plt.cm.tab10,
            vmin=0, vmax=9
        )


        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=1.5,
            arrows=self.directed,
            arrowsize=20
        )

        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def draw_graph(self):
        try:
            G = self.to_networkx()

            if not self.vertices:
                print("Graph is empty - no vertices to draw")
                return

            pos = nx.spring_layout(G, seed=42)

            plt.figure(figsize=(10, 8))
            nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue')
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

            if self.directed:
                nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20,
                                       edge_color='gray', width=1.5)
            else:
                nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5)

            edge_labels = nx.get_edge_attributes(G, 'weight')
            if edge_labels:
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

            plt.title("Graph Visualization", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error drawing graph: {e}")
            print("Vertices:", self.vertices)
            print("Edges:", self.edges)


if __name__ == "__main__":
    # Create a directed graph
    g = Graph(directed=True)
    g.add_edge('1', '2')
    g.add_edge('1', '3')
    g.add_edge('2', '4')
    g.add_edge('3', '2')
    g.add_edge('3', '4')
    g.add_edge('1', '5')
    g.add_edge('5', '3')
    g.add_edge('6', '4')
    g.add_edge('6', '3')


    # Draw all visualizations
    g.draw_graph()
    g.draw_adjacency_matrix()
    g.draw_incidence_matrix()
    g.show_successors_table()
    g.show_predecessors_table()
    g.draw_dynamic_list_of_vertices()
    g.draw_dynamic_list_of_arcs()
    g.draw_linear_list_of_successors()
    g.draw_linear_list_with_eol()
    g.draw_linear_list_with_bol()
    vertex_colors = g.greedy_vertex_coloring()
    edge_colors = g.greedy_edge_coloring()
    g.draw_vertex_coloring(vertex_colors)
    g.draw_edge_coloring(edge_colors)
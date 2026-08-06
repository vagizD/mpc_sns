import torch


class Boundary:
    def get_signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        pass


class CircleBoundary(Boundary):
    def __init__(
        self,
        center_x: float = 0.0,
        center_y: float = 0.0,
        radius: float = 0.5
    ):
        self.center = torch.tensor([[center_x, center_y]])
        self.radius = radius

    def get_signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        distance = (points - self.center.to(points.device)).square().sum(dim=-1).sqrt()
        is_outside = distance > self.radius

        distance_to_boundary = (distance - self.radius).abs()

        signed_distance = torch.where(is_outside, -distance_to_boundary, distance_to_boundary)
        return signed_distance


class StarBoundary(Boundary):
    def __init__(
        self,
        center_x: float = 0.0,
        center_y: float = 0.0,
        num_points: int = 5,
        inner_radius: float = 0.2,
        outer_radius: float = 0.5,
        initial_phase: float = torch.pi / 2
    ):
        self.center_x = center_x
        self.center_y = center_y
        self.num_points = num_points
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.initial_phase = initial_phase

        self.edges = self.build_edges()

    def build_edges(self) -> torch.Tensor:
        vertex_indices = torch.arange(2 * self.num_points)  # m = 2 * n - number of vertices

        radius = torch.where(vertex_indices % 2 == 0, self.outer_radius, self.inner_radius)

        phase_step = torch.pi / self.num_points
        phase = self.initial_phase + vertex_indices.float() * phase_step

        vertices = torch.stack(
            [radius * torch.cos(phase), radius * torch.sin(phase)],
        dim=-1)  # (m, 2)

        # add additional vertex to the end so that edges are build correctly
        vertices_unwrapped = torch.cat([vertices, vertices[0:]], dim=0)
        edges = torch.stack([vertices_unwrapped[:-1], vertices_unwrapped[1:]], dim=1)  # (m, 2, 2)
        return edges

    @staticmethod
    def get_projection(points: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        Projections of points onto edges.
        :param points: tensor of shape (n, 2), where n is number of points.
        :param edges: tensor of shape (m, 2, 2), where m is number of edges.
        :return: tensor of shape (n, m, 2), projection of each point on each edge.
        """
        start = edges[:, 0, :].unsqueeze(0)    # (1, m, 2), a
        end = edges[:, 1, :].unsqueeze(0)      # (1, m, 2), b
        direction = end - start                # (1, m, 2), u

        points_centered = points.unsqueeze(1) - start  # (n, 1, 2) - (1, m, 2) = (n, m, 2)

        # t_star = dot(p - a, u) / dot(u, u)  -- scale
        # q_star = a + t_star * u             -- projection of p on the line defined by point a and vector u
        # q      = a + clip(t_star, 0, 1) * u -- closest to p edge points
        scale = (points_centered * direction).sum(dim=2) / (direction * direction).sum(dim=2)  # (n, m),    t_star
        scale_clipped = torch.clip(scale, 0, 1).unsqueeze(2)                         # (n, m, 1), t
        projection = start + scale_clipped * direction                                         # (n, m, 2), q
        return projection

    @staticmethod
    def get_distance(points: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        """
        Returns distance from each point to the closest edge.
        :param points: tensor of shape (n, 2), where n is number of points.
        :param projection: tensor of shape (n, m, 2), where m is number of edges
            on which each point was projected.
        :return: tensor of shape (n,) - distances from each point to its closest edges
        """
        points = points.unsqueeze(1)                                 # (n, 1, 2)
        distance = (points - projection).square().sum(dim=2).sqrt()  # (n, m)
        return distance.min(dim=1).values                            # (n,)

    @staticmethod
    def get_signs(points: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        Returns signs for each point, -1 if the point lies outside the polygon defined by edges
            and 1 otherwise. The algorithm uses winding numbers to identify points locations.
        :param points: tensor of shape (n, 2), where n is number of points.
        :param edges: tensor of shape (m, 2, 2), where m is number of edges defining the polygon.
        :return: tensor of shape (n,) - signs for each point.
        """
        start = edges[:, 0, :].unsqueeze(0)    # (1, m, 2), a
        end = edges[:, 1, :].unsqueeze(0)      # (1, m, 2), b

        start_centered = start - points.unsqueeze(1)  # (1, m, 2) - (n, 1, 2) = (n, m, 2)
        end_centered = end - points.unsqueeze(1)                              # (n, m, 2)

        # In 2D case, cross product is defined as scalar (z-component of 3D cross product):
        # cross(a, b) = |a| * |b| * sin(alpha)
        # dot(a, b)   = |a| * |b| * cos(alpha)
        # atan2(cross(a, b), dot(a, b)) = atan2(sin(alpha), cos(alpha)) = alpha
        cross = StarBoundary.get_cross(start_centered, end_centered)  # (n, m)
        dot = (start_centered * end_centered).sum(dim=2)              # (n, m)
        alpha = torch.atan2(cross, dot)                               # (n, m)

        # (cumulative angle from walking through each edge) / (2 * pi) = number of turns
        winding_number = 1 / (2 * torch.pi) * alpha.sum(dim=1)        # (n,)

        # p is inside the polygon => 1 turn => w(p) = +1 or -1, depending on edges orientation
        # p is outside the polygon => 0 turns => w(p) = 0
        sign = torch.where(winding_number.abs() > 0.5, 1.0, -1.0)  # 0.5 to avoid calculation error
        return sign

    @staticmethod
    def get_cross(a: torch.Tensor, b: torch.Tensor):
        """
        Calculates cross product of 2D vectors.
        :param a: tensor of shape (..., 2), first vector.
        :param b: tensor of shape (..., 2), second vector.
        :return: tensor of shape (...), cross(a, b).
        """
        mult = a * b.flip(dims=(2,))  # (..., 2) -- [a1 * b2, a2 * b1]
        diff = mult[..., 0] - mult[..., 1]  # (...) -- [a1 * b2 - a2 * b1]
        return diff

    def get_signed_distance(self, points: torch.Tensor) -> torch.Tensor:
        projection = self.get_projection(points, self.edges)   # (n, m, 2)
        distance = self.get_distance(points, projection)
        sign = self.get_signs(points, self.edges)
        return sign * distance

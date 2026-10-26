import unittest
import torch
from torch import nn

from src import models


class ModelTests(unittest.TestCase):
    def test_teacher_has_no_maxpool(self):
        teacher = models.make_teacher(pretrained=False)
        self.assertIsInstance(teacher.maxpool, nn.Identity)
        # the pretrained layers keep their shapes: conv1 is still the 7x7 stride-2 convolution
        self.assertEqual(tuple(teacher.conv1.weight.shape), (64, 3, 7, 7))
        self.assertEqual(teacher(torch.zeros(2, 3, 64, 64)).shape, (2, 43))

    def test_student_output_shape(self):
        self.assertEqual(models.make_student()(torch.zeros(2, 3, 64, 64)).shape, (2, 43))


if __name__ == "__main__":
    unittest.main()

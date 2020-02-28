// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <map>
#include <memory>
#include <variant>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"
#include "sculpting_material.hpp"

namespace Sculptor {
class MatrixApplierBase;
class glObject;
class glInstancedObject;

class CubeSculptingMaterialCPU : public SculptingMaterial {
 public:
  CubeSculptingMaterialCPU(unsigned ncubes_per_side,
                           std::unique_ptr<MatrixApplierBase> matrix_applier);
  ~CubeSculptingMaterialCPU() override;

  void CollideWith(glObject& object) override;

 private:
  struct Node {
   public:
    void Subdivide(CubeSculptingMaterialCPU* material, glInstancedObject& tree);
    void Remove(CubeSculptingMaterialCPU* material);

    std::unique_ptr<Node> l = nullptr;
    std::unique_ptr<Node> r = nullptr;
    unsigned axis = 0;
    std::variant<unsigned, float> v = 0u;

   private:
    void RemoveChild(CubeSculptingMaterialCPU* material, Node* node);

    friend class CubeSculptingMaterialCPU;
    Node(std::unique_ptr<Node> left,
         std::unique_ptr<Node> right,
         unsigned axis,
         std::variant<unsigned, float> i);

    Node* parent_ = nullptr;

    Node(Node const&) = delete;
    Node& operator=(Node const&) = delete;
  };
  void OnNodeCreated(Node* node);
  void OnNodeDeleted(Node* node);
  bool ShouldRemoveNode(float distance);
  unsigned GetMaxDepth() { return max_depth_; }

  std::unique_ptr<Node> kd_tree_root_ = nullptr;
  std::map<unsigned, Node*> index_to_node_ = {};

  unsigned max_cubes_per_side;
  unsigned max_depth_;
};
}  // namespace Sculptor

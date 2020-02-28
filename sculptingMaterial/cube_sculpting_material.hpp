// Copyright 2020 Wojciech Replin. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <map>
#include <memory>
#include <variant>

#include "../cudaGraphics/cudaGraphicsResource/cuda_graphics_resource.hpp"

namespace Sculptor {
class MatrixApplierBase;
class glObject;
class glInstancedObject;

class CubeSculptingMaterial {
 public:
  struct Node {
   public:
    void Subdivide(CubeSculptingMaterial* material);
    void Remove(CubeSculptingMaterial* material);

    std::unique_ptr<Node> l = nullptr;
    std::unique_ptr<Node> r = nullptr;
    unsigned axis = 0;
    std::variant<unsigned, float> v = 0u;

   private:
    void RemoveChild(CubeSculptingMaterial* material, Node* node);

    friend class CubeSculptingMaterial;
    Node(std::unique_ptr<Node> left,
         std::unique_ptr<Node> right,
         unsigned axis,
         std::variant<unsigned, float> i);

    Node* parent_ = nullptr;

    Node(Node const&) = delete;
    Node& operator=(Node const&) = delete;
  };
  class CollisionAlgorithm {
   public:
    CollisionAlgorithm() = default;
    virtual ~CollisionAlgorithm() = default;
    virtual void Run(CubeSculptingMaterial* material, glObject& object) = 0;
    Node* SetTree(Node* tree);

    CollisionAlgorithm(CollisionAlgorithm&&) = default;
    CollisionAlgorithm& operator=(CollisionAlgorithm&&) = default;

   protected:
    Node* tree_ = nullptr;

   private:
    CollisionAlgorithm(CollisionAlgorithm const&) = delete;
    CollisionAlgorithm& operator=(CollisionAlgorithm const&) = delete;
  };

  CubeSculptingMaterial(
      unsigned ncubes_per_side,
      std::unique_ptr<glObject> reference_model,
      std::unique_ptr<MatrixApplierBase> matrix_applier,
      std::unique_ptr<CollisionAlgorithm> collision_algorithm);
  ~CubeSculptingMaterial();

  void Render(glm::mat4 const& vp);
  void Rotate(float amount);

  void Collide(glObject& object);

  glInstancedObject& GetObject();

  unsigned GetMaxDepth() { return max_depth_; }

  void OnNodeCreated(Node* node);
  void OnNodeDeleted(Node* node);
  bool ShouldRemoveNode(float distance);

 private:
  std::unique_ptr<Node> kd_tree_root_ = nullptr;
  std::map<unsigned, Node*> index_to_node_ = {};

  unsigned max_cubes_per_side;
  unsigned max_depth_;
  std::unique_ptr<glInstancedObject> visible_material_;
  std::unique_ptr<CollisionAlgorithm> collision_algorithm_;
};
}  // namespace Sculptor

// Copyright 2020 Wojciech Replin. All rights reserved.

#include "cube_sculpting_material.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <utility>
#include <vector>

#include "../glInstancedObject/gl_instanced_object.hpp"
#include "../glObject/gl_object.hpp"
#include "../matrixApplier/matrix_applier.hpp"

namespace Sculptor {
CubeSculptingMaterial::CubeSculptingMaterial(
    unsigned ncubes_per_side,
    std::unique_ptr<glObject> reference_model,
    std::unique_ptr<MatrixApplierBase> matrix_applier,
    std::unique_ptr<CollisionAlgorithm> collision_algorithm)
    : max_cubes_per_side(1 << ncubes_per_side),
      max_depth_(3 * ncubes_per_side),
      visible_material_(
          std::make_unique<glInstancedObject>(1 << (3 * ncubes_per_side),
                                              std::move(reference_model),
                                              std::move(matrix_applier))),
      collision_algorithm_(std::move(collision_algorithm)) {
  glm::vec3 initial_pos = {0.0, 0.0, 0.0};
  visible_material_->AddInstance(glm::translate(glm::mat4(1.f), initial_pos));
  kd_tree_root_.reset(new Node{nullptr, nullptr, 0, 0u});
  collision_algorithm_->SetTree(kd_tree_root_.get());
}

CubeSculptingMaterial::~CubeSculptingMaterial() = default;

void CubeSculptingMaterial::Render(glm::mat4 const& vp) {
  visible_material_->Render(vp * global_transform_);
}

void CubeSculptingMaterial::Rotate(float amount) {
  global_transform_ = glm::rotate(glm::mat4(1.f), amount, glm::vec3(0, 1, 0)) *
                      global_transform_;
}

void CubeSculptingMaterial::Collide(glObject& object) {
  object.Transform(glm::inverse(global_transform_));
  collision_algorithm_->Run(this, object);
  object.Transform(global_transform_);
}

glInstancedObject& CubeSculptingMaterial::GetObject() {
  return *visible_material_;
}

void CubeSculptingMaterial::OnNodeCreated(CubeSculptingMaterial::Node* node) {
  if (auto* i = std::get_if<unsigned>(&node->v))
    index_to_node_[*i] = node;
}

void CubeSculptingMaterial::OnNodeDeleted(CubeSculptingMaterial::Node* node) {
  if (auto* i = std::get_if<unsigned>(&node->v)) {
    auto it = index_to_node_.find(
        visible_material_->GetModelTransforms().GetSize() - 1);
    if (*i + 1 != visible_material_->GetModelTransforms().GetSize()) {
      auto m = visible_material_->GetTransformAt(
          visible_material_->GetModelTransforms().GetSize() - 1);
      auto* n = it->second;
      visible_material_->SetInstance(m, *i);
      n->v = *i;
      index_to_node_[*i] = n;
    }
    index_to_node_.erase(it);
    visible_material_->PopInstance();
  }
}

bool CubeSculptingMaterial::ShouldRemoveNode(float distance) {
  return distance <= 1.1f / max_cubes_per_side;
}

CubeSculptingMaterial::Node* CubeSculptingMaterial::CollisionAlgorithm::SetTree(
    CubeSculptingMaterial::Node* tree) {
  auto* ret = tree_;
  tree_ = tree;
  return ret;
}

CubeSculptingMaterial::Node::Node(
    std::unique_ptr<CubeSculptingMaterial::Node> left,
    std::unique_ptr<CubeSculptingMaterial::Node> right,
    unsigned ax,
    std::variant<unsigned, float> i)
    : l(std::move(left)), r(std::move(right)), axis(ax), v(i) {
  if (l)
    l->parent_ = this;
  if (r)
    r->parent_ = this;
}

void CubeSculptingMaterial::Node::Subdivide(CubeSculptingMaterial* material) {
  auto& tree = material->GetObject();
  auto m_original = tree.GetTransformAt(std::get<unsigned>(v));
  auto scale = glm::scale(glm::mat4(1.f), glm::vec3{0.5, 0.5, 0.5});
  auto middle = m_original * glm::vec4{0, 0, 0, 1};

  auto lll = m_original *
             glm::translate(glm::mat4(1.f), glm::vec3{-0.5, -0.5, -0.5}) *
             scale;
  auto rll = m_original *
             glm::translate(glm::mat4(1.f), glm::vec3{0.5, -0.5, -0.5}) * scale;
  auto lrl = m_original *
             glm::translate(glm::mat4(1.f), glm::vec3{-0.5, 0.5, -0.5}) * scale;
  auto llr = m_original *
             glm::translate(glm::mat4(1.f), glm::vec3{-0.5, -0.5, 0.5}) * scale;
  auto rrl = m_original *
             glm::translate(glm::mat4(1.f), glm::vec3{0.5, 0.5, -0.5}) * scale;
  auto rlr = m_original *
             glm::translate(glm::mat4(1.f), glm::vec3{0.5, -0.5, 0.5}) * scale;
  auto lrr = m_original *
             glm::translate(glm::mat4(1.f), glm::vec3{-0.5, 0.5, 0.5}) * scale;
  auto rrr = m_original *
             glm::translate(glm::mat4(1.f), glm::vec3{0.5, 0.5, 0.5}) * scale;

  std::unique_ptr<Node> lll_node(new Node{
      nullptr, nullptr, 0, tree.SetInstance(lll, std::get<unsigned>(v))}),
      rll_node(new Node{nullptr, nullptr, 0,
                        static_cast<unsigned>(tree.AddInstance(rll))}),
      lrl_node(new Node{nullptr, nullptr, 0,
                        static_cast<unsigned>(tree.AddInstance(lrl))}),
      llr_node(new Node{nullptr, nullptr, 0,
                        static_cast<unsigned>(tree.AddInstance(llr))}),
      rrl_node(new Node{nullptr, nullptr, 0,
                        static_cast<unsigned>(tree.AddInstance(rrl))}),
      rlr_node(new Node{nullptr, nullptr, 0,
                        static_cast<unsigned>(tree.AddInstance(rlr))}),
      lrr_node(new Node{nullptr, nullptr, 0,
                        static_cast<unsigned>(tree.AddInstance(lrr))}),
      rrr_node(new Node{nullptr, nullptr, 0,
                        static_cast<unsigned>(tree.AddInstance(rrr))});

  material->OnNodeCreated(lll_node.get());
  material->OnNodeCreated(rll_node.get());
  material->OnNodeCreated(lrl_node.get());
  material->OnNodeCreated(llr_node.get());
  material->OnNodeCreated(rrl_node.get());
  material->OnNodeCreated(rlr_node.get());
  material->OnNodeCreated(lrr_node.get());
  material->OnNodeCreated(rrr_node.get());

  std::unique_ptr<Node> rr_node(
      new Node{std::move(rrl_node), std::move(rrr_node), 2, middle.z}),
      rl_node(new Node{std::move(rll_node), std::move(rlr_node), 2, middle.z}),
      lr_node(new Node{std::move(lrl_node), std::move(lrr_node), 2, middle.z}),
      ll_node(new Node{std::move(lll_node), std::move(llr_node), 2, middle.z});

  material->OnNodeCreated(rr_node.get());
  material->OnNodeCreated(rl_node.get());
  material->OnNodeCreated(lr_node.get());
  material->OnNodeCreated(ll_node.get());

  std::unique_ptr<Node> l_node(
      new Node{std::move(ll_node), std::move(lr_node), 1, middle.y}),
      r_node(new Node{std::move(rl_node), std::move(rr_node), 1, middle.y});
  l_node->parent_ = this;
  r_node->parent_ = this;

  l = std::move(l_node);
  r = std::move(r_node);
  v = middle.x;
  axis = 0;

  material->OnNodeCreated(l.get());
  material->OnNodeCreated(r.get());
}

void CubeSculptingMaterial::Node::Remove(CubeSculptingMaterial* material) {
  parent_->RemoveChild(material, this);
}

void CubeSculptingMaterial::Node::RemoveChild(
    CubeSculptingMaterial* material,
    CubeSculptingMaterial::Node* node) {
  material->OnNodeDeleted(node);
  if (l.get() == node)
    l = nullptr;
  if (r.get() == node)
    r = nullptr;
  if (!r && !l && parent_)
    parent_->RemoveChild(material, this);
}
}  // namespace Sculptor

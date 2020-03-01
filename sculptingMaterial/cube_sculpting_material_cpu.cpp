// Copyright 2020 Wojciech Replin. All rights reserved.

#include "cube_sculpting_material_cpu.hpp"

#include <algorithm>
#include <functional>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <utility>
#include <vector>

#include "../glInstancedObject/gl_instanced_object.hpp"
#include "../glObject/gl_object.hpp"
#include "../matrixApplier/matrix_applier.hpp"
#include "../modelProvider/obj_provider.hpp"
#include "../shaderFactory/shader_factory.hpp"
#include "../shaderProgram/shader_program_base.hpp"
#include "../textureProvider/png_texture_provider.hpp"

namespace Sculptor {
namespace {
constexpr float kEps = 0.001;

float Len(glm::vec3 const& v) {
  return std::max(std::abs(v.x), std::max(std::abs(v.y), std::abs(v.z)));
}
}  // namespace
CubeSculptingMaterialCPU::CubeSculptingMaterialCPU(
    unsigned ncubes_per_side,
    ShaderFactory* shader_factory,
    std::unique_ptr<MatrixApplierBase> matrix_applier)
    : SculptingMaterial(std::make_unique<glInstancedObject>(
          1 << (3 * ncubes_per_side),
          std::make_unique<glObject>(
              std::make_unique<ObjProvider>("../Sculptor/model/cube.obj"),
              shader_factory->GetShader(ShaderFactory::ShaderType::PHONG,
                                        ShaderFactory::ObjectType::INSTANCED),
              std::make_unique<MatrixApplier>(),
              std::make_unique<PNGTextureProvider>(
                  "../Sculptor/texture/cube.png"),
              glm::vec4{1.0, 0.5, 1.0, 200.0}),
          std::move(matrix_applier))),
      max_cubes_per_side(1 << ncubes_per_side),
      max_depth_(3 * ncubes_per_side) {
  glm::vec3 initial_pos = {0.0, 0.0, 0.0};
  GetObject().AddInstance(glm::translate(glm::mat4(1.f), initial_pos));
  kd_tree_root_.reset(new Node{nullptr, nullptr, 0, 0u});
}

CubeSculptingMaterialCPU::~CubeSculptingMaterialCPU() = default;

void CubeSculptingMaterialCPU::CollideWith(glObject& object) {
  object.Transform(glm::inverse(GetObject().GetGlobalTransform()));

  Node* current_nearest_node = nullptr;
  float best_distance = std::numeric_limits<float>::infinity();
  glm::vec3 query_point, current_nearest;
  std::function<void(int, Node*)> algorithm;
  algorithm = [&algorithm, &query_point, &current_nearest_node, &best_distance,
               &current_nearest, material = this, this](int depth, Node* root) {
    if (auto* i = std::get_if<unsigned>(&root->v)) {
      if (depth > 0) {
        root->Subdivide(material, GetObject());
      } else {
        auto m = GetObject().GetTransformAt(*i);
        auto p = glm::vec3(m * glm::vec4{0, 0, 0, 1});
        auto dist = Len(query_point - p);
        if (best_distance > dist) {
          best_distance = dist;
          current_nearest = p;
          current_nearest_node = root;
        }
        return;
      }
    }
    auto diff = reinterpret_cast<float*>(&query_point)[root->axis] -
                std::get<float>(root->v);
    if (diff >= -kEps) {
      if (root->r)
        algorithm(depth - 1, root->r.get());
      if (root->l && best_distance > diff)
        algorithm(depth - 1, root->l.get());
    }
    if (diff <= kEps) {
      if (root->l)
        algorithm(depth - 1, root->l.get());
      if (root->r && best_distance > -diff)
        algorithm(depth - 1, root->r.get());
    }
  };
  for (auto& p : object.GetVertices()->ToStdVector()) {
    query_point = p;
    current_nearest_node = nullptr;
    best_distance = std::numeric_limits<float>::infinity();
    algorithm(GetMaxDepth(), kd_tree_root_.get());
    if (ShouldRemoveNode(best_distance))
      current_nearest_node->Remove(this);
  }

  object.Transform(GetObject().GetGlobalTransform());
}

void CubeSculptingMaterialCPU::OnNodeCreated(
    CubeSculptingMaterialCPU::Node* node) {
  if (auto* i = std::get_if<unsigned>(&node->v))
    index_to_node_[*i] = node;
}

void CubeSculptingMaterialCPU::OnNodeDeleted(
    CubeSculptingMaterialCPU::Node* node) {
  if (auto* i = std::get_if<unsigned>(&node->v)) {
    auto it =
        index_to_node_.find(GetObject().GetModelTransforms().GetSize() - 1);
    if (*i + 1 != GetObject().GetModelTransforms().GetSize()) {
      auto m = GetObject().GetTransformAt(
          GetObject().GetModelTransforms().GetSize() - 1);
      auto* n = it->second;
      GetObject().SetInstance(m, *i);
      n->v = *i;
      index_to_node_[*i] = n;
    }
    index_to_node_.erase(it);
    GetObject().PopInstance();
  }
}

bool CubeSculptingMaterialCPU::ShouldRemoveNode(float distance) {
  return distance <= 1.1f / max_cubes_per_side;
}

CubeSculptingMaterialCPU::Node::Node(
    std::unique_ptr<CubeSculptingMaterialCPU::Node> left,
    std::unique_ptr<CubeSculptingMaterialCPU::Node> right,
    unsigned ax,
    std::variant<unsigned, float> i)
    : l(std::move(left)), r(std::move(right)), axis(ax), v(i) {
  if (l)
    l->parent_ = this;
  if (r)
    r->parent_ = this;
}

void CubeSculptingMaterialCPU::Node::Subdivide(
    CubeSculptingMaterialCPU* material,
    glInstancedObject& tree) {
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

void CubeSculptingMaterialCPU::Node::Remove(
    CubeSculptingMaterialCPU* material) {
  parent_->RemoveChild(material, this);
}

void CubeSculptingMaterialCPU::Node::RemoveChild(
    CubeSculptingMaterialCPU* material,
    CubeSculptingMaterialCPU::Node* node) {
  material->OnNodeDeleted(node);
  if (l.get() == node)
    l = nullptr;
  if (r.get() == node)
    r = nullptr;
  if (!r && !l && parent_)
    parent_->RemoveChild(material, this);
}
}  // namespace Sculptor

/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LUCI_PARTITION_H__
#define __LUCI_PARTITION_H__

#include <luci/IR/Module.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace luci
{

/**
 * @brief PartitionTable holds partition information
 */
struct PartitionTable
{
  std::vector<std::string> groups;
  std::string default_group;

  // assign by opcode name: OPCODENAME=group
  std::unordered_map<std::string /* OPCODENAME */, std::string /* group */> byopcodes;

  // TODO add assign by OP name
};

/**
 * @brief PartedModule holds partitioned module and group name
 */
struct PartedModule
{
  std::unique_ptr<Module> module;
  // group name used to partition this module
  std::string group;

  // unique name(filename) of this module
  std::string name;
};

struct PartedModules
{
  std::vector<PartedModule> pmodules;

  // TODO add connections ?
};

/**
 * @brief Method to do paritioning from module and PartitionTable to produce PartedModules
 */
PartedModules apply(Module *module, const PartitionTable &partition);

} // namespace luci

#endif // __LUCI_PARTITION_H__

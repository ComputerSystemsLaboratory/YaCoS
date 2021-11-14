#!/bin/bash

function add_llvm_10_apt_source {
  curl https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
  if [[ $1 == "16.04" ]]; then
    echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-10 main" | sudo tee -a /etc/apt/sources.list
  elif [[ $1 == "18.04" ]]; then
    echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-10 main" | sudo tee -a /etc/apt/sources.list
  fi
  sudo apt-get -qq update
}

function install_system_packages {
  sudo apt install -y graphviz libgraphviz-dev
  sudo apt install -y libllvm10 llvm-10-dev
  sudo apt install -y clang-10 libclang1-10 libclang-10-dev libclang-common-10-dev
  sudo apt install -y creduce libeigen3-dev
  wget https://github.com/sharkdp/hyperfine/releases/download/v1.11.0/hyperfine_1.11.0_amd64.deb
  sudo dpkg -i hyperfine_1.11.0_amd64.deb
  rm hyperfine_1.11.0_amd64.deb
}

function install_yacos_data {
  mkdir $HOME/.local/yacos
  wget www.csl.uem.br/repository/data/yacos_data.tar.xz
  tar xfJ yacos_data.tar.xz -C $HOME/.local/yacos
  rm -f yacos_data.tar.xz
  wget www.csl.uem.br/repository/data/yacos_tests.tar.xz
  tar xfJ yacos_tests.tar.xz -C $HOME/.local/yacos
  rm -f yacos_tests.tar.xz
}

function install_ir2vec {
  git clone https://github.com/IITH-Compilers/IR2Vec.git
  cd IR2Vec
  git checkout llvm10
  mkdir build && cd build
  cmake -DLT_LLVM_INSTALL_DIR=/usr -DEigen3_DIR=/usr -DCMAKE_INSTALL_PREFIX=/usr ../src
  sudo make install
  cd ../..
  sudo rm /usr/seedEmbeddingVocab-300-llvm10.txt
  rm -rf IR2Vec
}

if [[ $(lsb_release -rs) == "16.04" ]] || [[ $(lsb_release -rs) == "18.04" ]]; then
  echo "OS supported."
  add_llvm_10_apt_source $(lsb_release -rs)
elif [[ $(lsb_release -rs) == "20.04" ]]; then
  echo "OS supported."
else
  echo "Non-supported OS. You have to install the packages manually."
  exit 1
fi

install_system_packages
install_yacos_data
install_ir2vec

echo "Please install perf, you may want to define instructions or cycles as the objective."

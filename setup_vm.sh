#!/bin/bash
set -e  # Exit on error

echo "====== GCP Tesla T4 Setup Script ======"
echo "This script will set up your GCP instance with NVIDIA drivers, PyTorch, and VS Code remote access capabilities."

# Update and install basic dependencies
echo "===== Updating system and installing dependencies ====="
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential cmake unzip pkg-config
sudo apt-get install -y libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
sudo apt-get install -y python3-dev python3-pip
sudo apt-get install -y openssh-server

# Set up the SSH server for VS Code remote access
echo "===== Configuring SSH server ====="
sudo systemctl enable ssh
sudo systemctl start ssh

# Install Git
echo "===== Installing and configuring Git ====="
sudo apt-get install -y git

# Set up Git credentials (will need to input manually)
echo "===== Setting up Git credentials ====="
echo "Please enter your Git username:"
read GIT_USERNAME
git config --global user.name "$GIT_USERNAME"

echo "Please enter your Git email:"
read GIT_EMAIL
git config --global user.email "$GIT_EMAIL"

# Generate SSH key for Git authentication
echo "===== Generating SSH key for Git ====="
ssh-keygen -t ed25519 -C "$GIT_EMAIL" -f ~/.ssh/id_ed25519 -N ""
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Display the public key to add to GitHub/GitLab
echo "===== Add this SSH key to your GitHub/GitLab account ====="
cat ~/.ssh/id_ed25519.pub
echo "Copy the above key and add it to your Git provider's SSH keys"
read -p "Press Enter to continue after adding the key..."

# Install NVIDIA drivers
echo "===== Installing NVIDIA drivers ====="
sudo apt-get install -y linux-headers-$(uname -r)
sudo apt-get install -y nvidia-driver-525  # Update version if needed

# Install CUDA toolkit (required for PyTorch)
echo "===== Installing CUDA toolkit ====="
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Install cuDNN (required for deep learning)
echo "===== Installing cuDNN ====="
sudo apt-get install -y libcudnn8 libcudnn8-dev

# Set up Python environment
echo "===== Setting up Python environment ====="
sudo pip3 install --upgrade pip
sudo pip3 install virtualenv

curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py -o install_gpu_driver.py
sudo python3 install_gpu_driver.py
# Create a virtual environment for PyTorch
echo "===== Creating Python virtual environment ====="
mkdir -p ~/pytorch_env
cd ~/pytorch_env
virtualenv venv
source venv/bin/activate

# Install PyTorch with CUDA support
echo "===== Installing PyTorch with CUDA support ====="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional useful data science packages
echo "===== Installing additional Python packages ====="
pip install numpy pandas matplotlib jupyter ipython

# Install VS Code server dependencies
echo "===== Installing VS Code server dependencies ====="
sudo apt-get install -y wget gpg apt-transport-https
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
rm -f packages.microsoft.gpg
sudo apt-get update

# Install additional useful tools
echo "===== Installing additional utilities ====="
sudo apt-get install -y htop tmux vim

# Verify NVIDIA driver installation
echo "===== Verifying NVIDIA installation ====="
nvidia-smi
nvcc --version

# Create a test PyTorch script to verify GPU access
echo "===== Creating PyTorch GPU test script ====="
cat > ~/test_gpu.py << 'EOL'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU device name:", torch.cuda.get_device_name(0))
    x = torch.rand(5, 3).cuda()
    print("Tensor on GPU:", x)
EOL

echo "===== Running PyTorch GPU test ====="
python3 ~/test_gpu.py

echo "====== Setup Complete ======"
echo "To connect via VS Code:"
echo "1. Install the 'Remote - SSH' extension in VS Code"
echo "2. Click on the green icon in the bottom-left corner"
echo "3. Select 'Connect to Host...'"
echo "4. Enter 'username@your-instance-ip'"
echo ""
echo "Your environment is ready for PyTorch development!"

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3-venv
while true; do
    read -p "Do you wish to setup up a vitual environment to install packages? (recommended) [Y/n] " yn
    case $yn in
        [Yy]* ) { 
                    python3 -m venv .env
                } || { 
                    python -m venv .env
                }; 
                source .env/bin/activate; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done
pip install -r requirements.txt
cat > run.sh <<EOL
Directory=".env"
if [ -d "\$Directory" ];
then
	source .env/bin/activate
fi
streamlit run app.py
EOL
chmod 755 run.sh
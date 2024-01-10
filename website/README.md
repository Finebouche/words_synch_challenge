## Lancement du backend en local

Pour lancer le backend en local, suivez les étapes suivantes :

1. Assurez-vous d'avoir Node.js installé sur votre machine.
2. Clonez le dépôt du backend depuis GitHub : `git clone https://github.com/votre-utilisateur/backend.git`
3. Accédez au répertoire du backend : `cd backend`
4. Installez les dépendances : `npm install`
5. Configurez les variables d'environnement nécessaires, telles que les informations de connexion à la base de données.
6. Lancez le serveur : `npm start`
7. Le backend sera accessible à l'adresse `http://localhost:3000`.

## Mise à jour du site

Pour mettre à jour le site en production, suivez les étapes suivantes :

1. Assurez-vous d'avoir accès au serveur de production.
2. Connectez-vous au serveur de production via SSH.
3. Accédez au répertoire du site : `cd /chemin/vers/le/site`
4. Mettez à jour le code en récupérant les dernières modifications depuis GitHub : `git pull origin master`
5. Installez les nouvelles dépendances : `npm install`
6. Redémarrez le serveur pour appliquer les changements : `npm restart`
7. Le site sera maintenant mis à jour et accessible aux utilisateurs.

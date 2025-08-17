Deploying Firestore rules & indexes

1. Install Firebase CLI
   npm install -g firebase-tools

2. Authenticate
   firebase login

3. Initialize (first time only)
   firebase init firestore
   - choose the project
   - when asked, point to infra/firestore/firestore.rules and infra/firestore/firestore.indexes.json

4. Deploy rules and indexes
   firebase deploy --only firestore:rules,firestore:indexes

Notes
- `firestore.indexes.json` is empty by default; add composite index definitions if needed.
- For CI: use a service account with Editor privileges and `firebase deploy --token "$FIREBASE_DEPLOY_TOKEN"`.
- Rules are enforced for client SDKs, Admin SDK bypasses rules.

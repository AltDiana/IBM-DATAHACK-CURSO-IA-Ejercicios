import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans


# ===========================================================
#   Clase Player
# ===========================================================
class Player:
    def __init__(self, player_name, character_type, avg_session_time,
                 matches_played, aggressive_actions, defensive_actions,
                 items_bought, victories, style=None):
        self.player_name = player_name
        self.character_type = character_type
        self.avg_session_time = avg_session_time
        self.matches_played = matches_played
        self.aggressive_actions = aggressive_actions
        self.defensive_actions = defensive_actions
        self.items_bought = items_bought
        self.victories = victories
        self.style = style  # aggressive o strategic


# ===========================================================
#   Clase GameModel
# ===========================================================
class GameModel:
    def __init__(self, players_list):
        self.players = players_list

        # Convertir jugadores a DataFrame
        self.data = pd.DataFrame([{
            "player_name": p.player_name,
            "character_type": p.character_type,
            "avg_session_time": p.avg_session_time,
            "matches_played": p.matches_played,
            "aggressive_actions": p.aggressive_actions,
            "defensive_actions": p.defensive_actions,
            "items_bought": p.items_bought,
            "victories": p.victories,
            "style": p.style
        } for p in players_list])

        # Codificador para character_type y style
        self.encoder_character = LabelEncoder()
        self.encoder_style = LabelEncoder()

        # Modelos
        self.class_model = None
        self.regression_model = None
        self.cluster_model = None

    # -------------------------------------------------------
    #   Modelo de Clasificación
    # -------------------------------------------------------
    def train_classification_model(self):
        df = self.data.copy()

        df["character_type_encoded"] = self.encoder_character.fit_transform(df["character_type"])
        df["style_encoded"] = self.encoder_style.fit_transform(df["style"])

        features = ["character_type_encoded", "avg_session_time", "matches_played",
                    "aggressive_actions", "defensive_actions", "items_bought"]

        X = df[features]
        y = df["style_encoded"]

        self.class_model = LogisticRegression()
        self.class_model.fit(X, y)

    def predict_style(self, player):
        row = pd.DataFrame([{
            "character_type_encoded": self.encoder_character.transform([player.character_type])[0],
            "avg_session_time": player.avg_session_time,
            "matches_played": player.matches_played,
            "aggressive_actions": player.aggressive_actions,
            "defensive_actions": player.defensive_actions,
            "items_bought": player.items_bought
        }])

        pred = self.class_model.predict(row)[0]
        return self.encoder_style.inverse_transform([pred])[0]

    # -------------------------------------------------------
    #   Modelo de Regresión
    # -------------------------------------------------------
    def train_regression_model(self):
        df = self.data.copy()

        df["character_type_encoded"] = self.encoder_character.fit_transform(df["character_type"])

        features = ["character_type_encoded", "avg_session_time", "matches_played",
                    "aggressive_actions", "defensive_actions", "items_bought"]

        X = df[features]
        y = df["victories"]

        self.regression_model = LinearRegression()
        self.regression_model.fit(X, y)

    def predict_victories(self, player):
        row = pd.DataFrame([{
            "character_type_encoded": self.encoder_character.transform([player.character_type])[0],
            "avg_session_time": player.avg_session_time,
            "matches_played": player.matches_played,
            "aggressive_actions": player.aggressive_actions,
            "defensive_actions": player.defensive_actions,
            "items_bought": player.items_bought
        }])

        return self.regression_model.predict(row)[0]

    # -------------------------------------------------------
    #   Modelo de Clustering
    # -------------------------------------------------------
    def train_clustering_model(self, n_clusters=2):
        df = self.data.copy()

        df["character_type_encoded"] = self.encoder_character.fit_transform(df["character_type"])

        features = ["character_type_encoded", "avg_session_time", "matches_played",
                    "aggressive_actions", "defensive_actions", "items_bought"]

        X = df[features]

        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.data["cluster"] = self.cluster_model.fit_predict(X)

    def assign_cluster(self, player):
        row = pd.DataFrame([{
            "character_type_encoded": self.encoder_character.transform([player.character_type])[0],
            "avg_session_time": player.avg_session_time,
            "matches_played": player.matches_played,
            "aggressive_actions": player.aggressive_actions,
            "defensive_actions": player.defensive_actions,
            "items_bought": player.items_bought
        }])

        return int(self.cluster_model.predict(row)[0])

    # -------------------------------------------------------
    #   OPCIONAL: Mostrar jugadores por cluster
    # -------------------------------------------------------
    def show_players_by_cluster(self):
        if "cluster" not in self.data.columns:
            print("Debes entrenar primero el modelo de clustering.")
            return

        for cluster_id in sorted(self.data["cluster"].unique()):
            print(f"\nCluster {cluster_id}:")
            cluster_players = self.data[self.data["cluster"] == cluster_id]

            for _, row in cluster_players.iterrows():
                print(f"{row['player_name']} - {row['character_type'].capitalize()} - {row['style'].capitalize()}")


# ===========================================================
#   EJEMPLO DE USO REAL
# ===========================================================
if __name__ == "__main__":
    # Datos de ejemplo
    players_data = [
        Player("P1", "mage", 40, 30, 90, 50, 20, 18, "aggressive"),
        Player("P2", "tank", 60, 45, 50, 120, 25, 24, "strategic"),
        Player("P3", "archer", 50, 35, 95, 60, 22, 20, "aggressive"),
        Player("P4", "tank", 55, 40, 60, 100, 28, 22, "strategic"),
    ]

    model = GameModel(players_data)
    model.train_classification_model()
    model.train_regression_model()
    model.train_clustering_model(n_clusters=2)

    # Jugador nuevo
    new_player = Player("TestPlayer", "mage", 42, 33, 88, 45, 21, 0)

    print("\n--- RESULTADOS ---")
    print("Estilo predicho:", model.predict_style(new_player))
    print("Victorias predichas:", model.predict_victories(new_player))
    print("Cluster asignado:", model.assign_cluster(new_player))

    # Mostrar clusters
    print("\n--- JUGADORES POR CLUSTER ---")
    model.show_players_by_cluster()

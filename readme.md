## 📊 Estructura de Base de Datos

Basándonos en los requerimientos del sistema, se necesitan al menos **dos tablas principales** para gestionar el acceso web y almacenar los análisis de commits generados por el sistema.

---

## 1️⃣ Tabla: `users` (Usuarios)

El sistema requiere un **login**, mediante el cual cada usuario podrá acceder al dashboard y visualizar la información relacionada con su actividad dentro del repositorio.

### Campos

| Campo | Tipo | Descripción |
|------|------|-------------|
| `id` | INT / UUID | Identificador único del usuario |
| `username` / `email` | VARCHAR | Credenciales utilizadas para el acceso al sistema |
| `password` | VARCHAR | Contraseña almacenada de forma segura mediante hash |
| `created_at` | DATETIME | Fecha en la que se registró el usuario |

Esta tabla permite **gestionar la autenticación y el acceso al sistema**, asegurando que cada usuario tenga sus propios registros y análisis asociados.

---

## 2️⃣ Tabla: `commit_analyses` (Análisis de Commits)

Cada vez que el sistema analiza un **commit**, los resultados se guardan en esta tabla. Esto permite mantener un **historial completo de cambios analizados** y sus respectivos niveles de riesgo.

### Campos

| Campo | Tipo | Descripción |
|------|------|-------------|
| `id` | INT / UUID | Identificador único del registro |
| `commit_hash` | VARCHAR | Identificador único del commit en el repositorio |
| `user_id` | INT / UUID | Relación con el usuario que realizó el commit |
| `risk_score` | INT | Puntaje de riesgo del commit (0 a 100) |
| `lines_added` | INT | Número de líneas agregadas en el commit |
| `lines_deleted` | INT | Número de líneas eliminadas |
| `files_modified` | INT | Cantidad de archivos modificados |
| `tests_modified` | BOOLEAN | Indica si se modificaron archivos de pruebas |
| `complex_sql_added` | BOOLEAN | Indica si se detectaron consultas SQL complejas |
| `analysis_date` | DATETIME | Fecha en la que se realizó el análisis |

---

## 🔗 Relación entre tablas

- Un **usuario (`users`)** puede tener **muchos análisis de commits (`commit_analyses`)**.
- La relación se establece mediante el campo **`user_id`** en la tabla `commit**_**

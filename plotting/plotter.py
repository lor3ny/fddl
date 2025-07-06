import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import sys

def DrawCatPlot(data, name):
    print(f"Plotting data collective: {name}")

    # Stile elegante per paper
    sns.set_theme(style="whitegrid", context="paper", font_scale=4,)

    # Conversione in DataFrame
    df = pd.DataFrame(data)

    # Ordina batch se sono numeri
    try:
        df["Batch Size"] = pd.Categorical(
            df["Batch Size"],
            categories=sorted(df["Batch Size"].unique(), key=lambda x: int(x)),
            ordered=True
        )
    except:
        pass  # Se non si può convertire in int, salta

    # Palette calda con sfumature di rosso
    custom_palette = ["#D91818", "#D97904", "#F2B705"]
    palette = custom_palette[:len(df["Type"].unique())]

    # Crea catplot
    g = sns.catplot(
        data=df,
        kind="bar",
        x="Batch Size",
        y="Latency",
        hue="Type",
        errorbar=("sd", 4),  # Aumenta visibilità dell'errore standard (2x)
        palette=palette,
        alpha=0.9,
        height=8,
        aspect=2,
        edgecolor="black",  # Bordi alle barre
        linewidth=1
    )

    # Etichette e titolo
    g.set_axis_labels("Batch Size", "Latency (ms)", labelpad=15)
    g.set_titles(f"{name}")
    g.set_xticklabels(rotation=0)

    # for ax in g.axes.flat:
    #     ax.tick_params(axis='both', labelsize=23)  # ad esempio: 14 pt

    g._legend.remove()

    # Aggiunta bordo al grafico
    for ax in g.axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)

    # Salvataggio
    g.figure.tight_layout()
    g.figure.savefig(f"{name}_catplot.png", dpi=300, bbox_inches='tight')
    plt.close()

# def DrawCatPlot(data, name):
#     print(f"Plotting data collective: {name}")

#     # Stile elegante per paper
#     sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

#     # Conversione in DataFrame
#     df = pd.DataFrame(data)

#     # Ordina batch se sono numeri
#     try:
#         df["Batch Size"] = pd.Categorical(
#             df["Batch Size"],
#             categories=sorted(df["Batch Size"].unique(), key=lambda x: int(x)),
#             ordered=True
#         )
#     except:
#         pass  # Se non si può convertire in int, salta

#     # Palette specifica e coerente
#     types = df["Type"].unique()
#     palette = sns.color_palette("Set2", n_colors=len(types))

#     # Crea catplot
#     g = sns.catplot(
#         data=df,
#         kind="bar",
#         x="Batch Size",
#         y="Latency",
#         hue="Type",
#         errorbar="sd",
#         palette=palette,
#         alpha=0.8,
#         height=6,
#         aspect=2  # figura larga
#     )

#     # Migliora asse
#     g.set_axis_labels("Batch Size", "Latency (ms)", labelpad=10)
#     g.set_titles(f"{name}")
#     g._legend.set_title("Type")

#     # Migliora tick
#     g.set_xticklabels(rotation=0)

#     # Salvataggio
#     g.figure.tight_layout()
#     g.figure.savefig(f"{name}_catplot.png", dpi=300, bbox_inches='tight')
#     plt.close()

def DrawLinePlot(data, name):
    print(f"Plotting data collective: {name}")

    # Imposta stile e contesto
    sns.set_style("whitegrid")
    sns.set_context("talk")

    # Crea figura
    f, ax1 = plt.subplots(figsize=(20, 10))

    # Conversione dati in DataFrame
    df = pd.DataFrame(data)
    df['cluster_collective'] = df['Cluster'].astype(str) + '_' + df['collective'].astype(str)

    # Palette migliorata
    palette = sns.color_palette("Set2", n_colors=df['cluster_collective'].nunique())

    # Lineplot con stile e markers
    sns.lineplot(
        data=df,
        x='message',
        y='bandwidth',
        hue='cluster_collective',
        style='cluster_collective',
        markers=True,
        markersize=10,
        linewidth=3,
        ax=ax1,
        palette=palette
    )

    # Linea teorica
    ax1.axhline(
        y=100,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Theoretical Peak {100} Gb/s'
    )

    # Etichette
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_ylabel('Bandwidth (Gb/s)', fontsize=28, labelpad=20)
    ax1.set_xlabel('Message Size', fontsize=28, labelpad=20)
    ax1.set_title(f'{name}', fontsize=38, pad=30)

    # Legenda centrata in basso fuori dal grafico
    ax1.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=3, frameon=True)

    plt.tight_layout()

    # Salvataggio figura
    plt.savefig(f'plots/{name}_line.png', dpi=300, bbox_inches='tight')
    plt.close()



def DrawScatterPlot(data, name):
    print(f"Plotting data collective: {name}")

    # Use a dark theme for the plot
    sns.set_style("whitegrid")  # darker background for axes
    sns.set_context("talk")

    # Create the figure and axes
    f, ax1 = plt.subplots(figsize=(20, 10))
    
    # Convert input data to a DataFrame
    df = pd.DataFrame(data)
    df['cluster_collective'] = df['Cluster'].astype(str) + '_' + df['collective'].astype(str)
    palette = sns.color_palette("Set2", n_colors=df['cluster_collective'].nunique())

    # Plot with seaborn
    fig = sns.scatterplot(
        data=df,
        x='iteration',
        y='bandwidth',
        hue='Cluster',
        style='Message',
        s=80,
        ax=ax1,
        alpha=0.9,
        palette=palette
    )

    # Labeling and formatting
    ax1.tick_params(axis='both', which='major', labelsize=28)
    ax1.set_ylabel('Bandwidth (Gb/s)', fontsize=35, labelpad=20)
    ax1.set_xlabel('Iterations', fontsize=35, labelpad=20)
    ax1.set_title(f'{name}', fontsize=45, pad=30)

    # Show legend and layout
    # Filtra legenda: solo cluster_collective unici + linea teorica

    ax1.legend(
        fontsize=25,           # grandezza testo etichette
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),  # più spazio sotto
        ncol=5,
        frameon=True,
        title=None,
        markerscale=2.0        # ingrandisce i marker nella legenda
    )
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'plots/{name}_scatter.png', dpi=300)  # save with dark background



def LoadData(data, path, type='default', mem=1, batch=0):

    latencies = []
    with open(path, 'r') as file:
        lines = file.readlines()[2:]  # Skip the first line
        for line in lines:
            latency = float(line.strip())
            latencies.append(latency)

    data['Latency'].extend(latencies)
    data['Type'].extend([type]*len(latencies))
    data['Batch Size'].extend([batch]*len(latencies))

    return data


def CleanData(data):
    for key in data.keys():
        data[key] = []
    return data

if __name__ == "__main__":


    data = {
        'Latency': [],
        'Type': [],
        'Batch Size': []
    }


    batch_size = [128,256,512]

    for i in range(3):
        data = LoadData(data, f"Single_GPU_{batch_size[i]}.csv", type="Single GPU", mem=64, batch=batch_size[i])
        data = LoadData(data, f"Data_Parallel_{batch_size[i]}.csv", type="Data Parallel", batch=batch_size[i])
        data = LoadData(data, f"Batch_Tensor_Parallel_MPI_{batch_size[i]}.csv", type="Data Tensor Parallel", mem=64*4, batch=batch_size[i])
    
    DrawCatPlot(data, f"Benchmark")


    CleanData(data)

    # for coll in collectives:
    #     for mess in messages:  
    #         data = LoadData(data, "HAICGU", nodes , folder_haicgu, [mess], cong=False, coll=coll)
    #         data = LoadData(data, "Nanjing", nodes , folder_nanjing, [mess], cong=False, coll=coll)
    #         DrawScatterPlot(data, f"Nanjing vs HAICGU {nodes} Nodes {coll} {mess}")
    #         CleanData(data)

    # for coll in collectives:
    #     data = LoadData(data, "HAICGU", nodes , folder_haicgu, messages=messages, cong=False, coll=coll)
    #     data = LoadData(data, "Nanjing", nodes , folder_nanjing, messages=messages, cong=False, coll=coll)
    #     DrawLinePlot(data, f"Nanjing vs HAICGU {nodes} Nodes {coll}")
    #     CleanData(data)

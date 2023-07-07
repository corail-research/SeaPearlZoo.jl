function plot_ppo_losses(csv_path::AbstractString)
    df = CSV.read(csv_path, DataFrame)

    instance_ids = unique(df.num_instance)

    figs = [plot() for _ in instance_ids]

    for (i,inst) in enumerate(instance_ids)
        instance = df[df.num_instance .== inst, :]

        losses = ["actor_loss", "critic_loss", "entropy_loss", "total_loss"]

        for loss in losses
            x = instance.num_update
            y = instance[!,loss]
            plot!(figs[i], x, y, label = loss)  
            title!(figs[i], string("PPO Losses for instance ", i))
        end
    end

    for fig in figs
        display(fig)
    end
end